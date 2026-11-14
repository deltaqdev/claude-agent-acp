import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { getSessionMessages } from "@anthropic-ai/claude-agent-sdk";
import { randomUUID } from "node:crypto";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { readHistoryAcrossCompactions } from "../resumed-session.js";
import { parseRewindSessionRequest, planRewind } from "../session-rewind.js";
import { encodeProjectPath } from "../session-index/project-dirs.js";

// Use the real SDK against temporary transcripts, without starting Claude Code.
let root: string;
let cwd: string;
let sessionId: string;
let ids: Record<string, string>;

/** Writes `records` as the transcript of `sessionId`, linking each to the one
 *  before it unless it names its own `parent` key. */
async function writeTranscript(
  records: Array<{ key: string; parent?: string | null; [field: string]: unknown }>,
) {
  ids = Object.fromEntries(records.map(({ key }) => [key, randomUUID()]));
  const lines = records.map(({ key, parent, ...record }, index) => {
    const previous = index > 0 ? records[index - 1].key : null;
    const parentKey = parent === undefined ? previous : parent;
    return JSON.stringify({
      sessionId,
      cwd,
      isSidechain: false,
      timestamp: new Date(1_700_000_000_000 + index).toISOString(),
      uuid: ids[key],
      parentUuid: parentKey === null ? null : ids[parentKey],
      ...record,
    });
  });
  const projectDir = path.join(root, "config", "projects", encodeProjectPath(cwd));
  await fs.mkdir(projectDir, { recursive: true });
  await fs.writeFile(path.join(projectDir, `${sessionId}.jsonl`), lines.join("\n") + "\n");
}

/** Appends `records` as they are: Claude Code's checkpoints carry no uuid. */
async function appendRecords(records: object[]) {
  const file = path.join(root, "config", "projects", encodeProjectPath(cwd), `${sessionId}.jsonl`);
  await fs.appendFile(file, records.map((record) => JSON.stringify(record) + "\n").join(""));
}

/** The checkpoints Claude Code's `rewind_conversation` appends, keeping `leaf` and dropping `cut`. */
function rewindCheckpoints(leaf: string, cut: string) {
  return [
    { type: "last-prompt", leafUuid: leaf, explicit: true, rewound: true, cutUuid: cut, sessionId },
    { type: "last-prompt", lastPrompt: "Fourth prompt", leafUuid: leaf, sessionId },
    { type: "cost-state", sessionId },
  ];
}

const prompt = (key: string, content: string) => ({
  key,
  type: "user",
  message: { role: "user", content },
});
const reply = (key: string, id: string, content: object[]) => ({
  key,
  type: "assistant",
  message: { id, role: "assistant", content },
});
const text = (value: string) => ({ type: "text", text: value });
const attachment = (key: string) => ({ key, type: "attachment", attachment: { type: "todo" } });

beforeEach(async () => {
  root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), "session-rewind-")));
  cwd = path.join(root, "project");
  await fs.mkdir(cwd);
  vi.stubEnv("CLAUDE_CONFIG_DIR", path.join(root, "config"));
  sessionId = randomUUID();
});

afterEach(async () => {
  vi.unstubAllEnvs();
  await fs.rm(root, { recursive: true, force: true });
});

describe("planRewind", () => {
  beforeEach(async () => {
    await writeTranscript([
      prompt("p1", "First prompt"),
      reply("r1", "msg_one", [text("First answer")]),
      attachment("t1"),
      prompt("p2", "Second prompt"),
      attachment("a2"),
      reply("r2a", "msg_two", [
        text("Reading"),
        { type: "tool_use", id: "toolu_1", name: "Read", input: {} },
      ]),
      {
        key: "tr2",
        type: "user",
        message: {
          role: "user",
          content: [{ type: "tool_result", tool_use_id: "toolu_1", content: "file" }],
        },
      },
      reply("r2b", "msg_three", [text("Second answer")]),
      prompt("p3", "Third prompt"),
      reply("r3", "msg_four", [text("Partial")]),
      prompt("m3", "[Request interrupted by user]"),
      prompt("p4", "Fourth prompt"),
      reply("r4", "msg_five", [text("Fourth answer")]),
    ]);
  });

  it.each([
    { name: "a reply", to: "msg_one", cut: "t1", drops: undefined, firstDiscarded: "p2" },
    { name: "a prompt", to: "p2", cut: "a2", drops: undefined, firstDiscarded: "r2a" },
    {
      name: "a reply before a tool result",
      to: "msg_two",
      cut: "tr2",
      drops: undefined,
      firstDiscarded: "r2b",
    },
    {
      name: "an interrupted reply",
      to: "msg_four",
      cut: "m3",
      drops: "p4",
      firstDiscarded: "p4",
    },
  ])(
    "keeps what belongs with $name and resumes at its last record",
    async ({ to, cut, drops, firstDiscarded }) => {
      const toMessageId = to.startsWith("msg_") ? to : ids[to];
      const plan = await planRewind(sessionId, toMessageId);
      expect(plan?.cut).toEqual({
        resumeSessionAt: ids[cut],
        ...(drops ? { resumeDropsTurn: ids[drops] } : {}),
      });
      expect(plan?.firstDiscarded).toBe(ids[firstDiscarded]);
      expect(plan?.firstDiscardedIsAssistant).toBe(firstDiscarded.startsWith("r"));
      expect(plan?.lastPrompt).toBe(ids.p4);
    },
  );

  it("maps the kept message ids to their uuids", async () => {
    const plan = await planRewind(sessionId, "msg_three");
    expect(Object.fromEntries(plan!.keptMessageIds)).toEqual({
      [ids.p1]: ids.p1,
      msg_one: ids.r1,
      [ids.p2]: ids.p2,
      msg_two: ids.r2a,
      [ids.tr2]: ids.tr2,
      msg_three: ids.r2b,
    });
  });

  it("changes nothing when no message follows", async () => {
    await expect(planRewind(sessionId, "msg_five")).resolves.toBeUndefined();
  });

  it.each([
    { name: "an id the session never sent", to: "msg_missing", firstDiscarded: undefined },
    { name: "a message a restart left out", to: "msg_five", firstDiscarded: "p3" },
  ])("refuses $name as unknown", async ({ to, firstDiscarded }) => {
    await expect(
      planRewind(sessionId, to, firstDiscarded && ids[firstDiscarded]),
    ).rejects.toMatchObject({
      code: -32002,
      data: { reason: "unknown_message_id", sessionId, messageId: to },
    });
  });

  it("reads the history an in-place rewind kept", async () => {
    await appendRecords(rewindCheckpoints(ids.r2b, ids.p3));

    await expect(planRewind(sessionId, "msg_five")).rejects.toMatchObject({
      data: { reason: "unknown_message_id" },
    });
    await expect(planRewind(sessionId, "msg_three")).resolves.toBeUndefined();
    expect((await planRewind(sessionId, "msg_one"))?.lastPrompt).toBe(ids.p2);
  });
});

describe("the replay after a rewind", () => {
  beforeEach(async () => {
    await writeTranscript([
      prompt("p1", "First prompt"),
      reply("r1", "msg_one", [text("First answer")]),
      prompt("p2", "Second prompt"),
      reply("r2", "msg_two", [text("Second answer")]),
    ]);
  });

  async function replayed() {
    const messages = await readHistoryAcrossCompactions(
      sessionId,
      await getSessionMessages(sessionId),
    );
    return messages.map((message) => message.uuid);
  }

  it("ends at the message the rewind kept", async () => {
    await appendRecords(rewindCheckpoints(ids.r1, ids.p2));
    expect(await replayed()).toEqual([ids.p1, ids.r1]);
  });

  it.each([
    {
      name: "a prompt written after the rewind",
      after: (leaf: string) => [
        ...rewindCheckpoints(leaf, ids.p2),
        {
          type: "user",
          uuid: randomUUID(),
          parentUuid: leaf,
          sessionId,
          cwd,
          message: { role: "user", content: "Third prompt" },
        },
      ],
      length: 3,
    },
    {
      name: "a checkpoint that is not a rewind",
      after: (leaf: string) => [{ type: "last-prompt", leafUuid: leaf, sessionId }],
      length: 4,
    },
    {
      name: "a later checkpoint of another leaf",
      after: (leaf: string) => [
        ...rewindCheckpoints(leaf, ids.p2),
        { type: "last-prompt", leafUuid: ids.r2, sessionId },
      ],
      length: 4,
    },
  ])("follows the newest message after $name", async ({ after, length }) => {
    await appendRecords(after(ids.r1));
    expect(await replayed()).toHaveLength(length);
  });
});

it("refuses every id of a session with no transcript as unknown", async () => {
  await expect(planRewind(randomUUID(), "msg_one")).rejects.toMatchObject({
    code: -32002,
    data: { reason: "unknown_message_id" },
  });
});

it("cuts at the kept message when a compaction relinked the next one", async () => {
  // The SDK moves the preserved segment p1..r1 after the summary, and links p2,
  // written after the summary, to its tail r1.
  await writeTranscript([
    prompt("old", "Old prompt"),
    reply("oldReply", "msg_old", [text("Old answer")]),
    prompt("p1", "First prompt"),
    reply("r1", "msg_one", [text("First answer")]),
    {
      key: "boundary",
      parent: null,
      type: "system",
      subtype: "compact_boundary",
      compactMetadata: { trigger: "manual", preservedSegment: {} },
    },
    {
      key: "summary",
      type: "user",
      isCompactSummary: true,
      message: { role: "user", content: "Summary of the old prompt" },
    },
    prompt("p2", "Second prompt"),
    reply("r2", "msg_two", [text("Second answer")]),
  ]);
  const file = path.join(root, "config", "projects", encodeProjectPath(cwd), `${sessionId}.jsonl`);
  const records = (await fs.readFile(file, "utf8"))
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line));
  records[4].compactMetadata.preservedSegment = {
    headUuid: ids.p1,
    anchorUuid: ids.summary,
    tailUuid: ids.r1,
  };
  await fs.writeFile(file, records.map((record) => JSON.stringify(record)).join("\n") + "\n");

  const plan = await planRewind(sessionId, "msg_one");

  expect(plan?.cut).toEqual({ resumeSessionAt: ids.r1, resumeDropsTurn: ids.p2 });
});

describe("parseRewindSessionRequest", () => {
  it.each([
    ["not an object", "x"],
    ["no sessionId", { toMessageId: "msg_one" }],
    ["no toMessageId", { sessionId: "s" }],
    ["an empty toMessageId", { sessionId: "s", toMessageId: "" }],
  ])("rejects %s", (_name, params) => {
    expect(() => parseRewindSessionRequest(params)).toThrow(
      expect.objectContaining({ code: -32602 }),
    );
  });

  it("keeps the session and message ids", () => {
    expect(parseRewindSessionRequest({ sessionId: "s", toMessageId: "msg_one", extra: 1 })).toEqual(
      { sessionId: "s", toMessageId: "msg_one" },
    );
  });
});
