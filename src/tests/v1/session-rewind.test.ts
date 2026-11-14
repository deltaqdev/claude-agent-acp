import { appendFile, mkdir, mkdtemp, realpath, rm, writeFile } from "node:fs/promises";
import { randomUUID } from "node:crypto";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ModelInfo, Options } from "@anthropic-ai/claude-agent-sdk";
import { ClaudeAcpAgent } from "../../v1/acp-agent.js";
import type { AcpClient } from "../../acp-client.js";
import { encodeProjectPath } from "../../session-index/project-dirs.js";
import { initializeClient, makeMockQuery } from "../helpers.js";
import { successfulResultMessage, userEcho } from "./session-doubles.js";

const sdk = vi.hoisted(() => ({ query: vi.fn() }));
vi.mock("@anthropic-ai/claude-agent-sdk", async () => ({
  ...(await vi.importActual<typeof import("@anthropic-ai/claude-agent-sdk")>(
    "@anthropic-ai/claude-agent-sdk",
  )),
  ...sdk,
  resolveSettings: async () => ({ effective: {}, sources: [], provenance: {} }),
  filterEscalatingDefaultMode: (result: any) => result.effective,
}));

const MODELS: ModelInfo[] = ["sonnet", "opus"].map((value) => ({
  value,
  displayName: value,
  description: "",
  resolvedModel: `claude-${value}-4-6`,
  supportsAutoMode: true,
}));

/** The CLI's error result for a cut that `--resume-drops-turn` refuses. */
const DROPS_TURN_REFUSAL =
  "Claude Code returned an error result: Resume rejected by --resume-drops-turn: resuming at x would discard entries not attributable to turn y";

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => (resolve = done));
  return { promise, resolve };
}

describe("_sh.rexa/session/rewind", { timeout: 10_000 }, () => {
  let agent: ClaudeAcpAgent;
  let root: string;
  let cwd: string;
  let transcript: string;
  let queries: ReturnType<typeof createQuery>[];
  let ids: Record<string, string>;
  let initializationGate: Promise<void> | undefined;
  let refuseCut: boolean;
  /** Claude Code's answer to `rewind_conversation`. */
  let answerRewind: (request: Record<string, any>) => Promise<unknown>;

  function createQuery(request: { options: Options; prompt: AsyncIterable<any> }) {
    const state = { model: request.options.model ?? "sonnet" };
    const prompts: string[] = [];
    const query = {
      ...makeMockQuery(),
      request: vi.fn((control: Record<string, any>) => answerRewind(control)),
      options: request.options,
      state,
      prompts,
      initializationResult: async () => {
        await initializationGate;
        if (refuseCut && request.options.resumeSessionAt) throw new Error(DROPS_TURN_REFUSAL);
        return { models: MODELS };
      },
      setModel: vi.fn(async (value: string) => {
        state.model = value;
      }),
      setPermissionMode: vi.fn(async () => {}),
      close: vi.fn(),
      [Symbol.asyncIterator]: async function* () {
        for await (const message of request.prompt) {
          prompts.push(message.message.content[0]?.text ?? message.message.content);
          yield userEcho(message);
          yield successfulResultMessage();
          yield { type: "system", subtype: "status", status: "idle" };
        }
      },
    };
    const iterator = query[Symbol.asyncIterator]();
    return Object.assign(query, { next: iterator.next.bind(iterator) });
  }

  /** As Claude Code answers a rewind before the second prompt: it keeps the
   *  first reply as the leaf and appends its checkpoint. */
  async function rewindInPlace(control: Record<string, any>) {
    await appendFile(
      transcript,
      JSON.stringify({
        type: "last-prompt",
        leafUuid: ids.r1,
        explicit: true,
        rewound: true,
        cutUuid: control.target_message_uuid,
      }) + "\n",
    );
    return { subtype: "success", response: { rewound: true, precedingAssistantUuid: ids.r1 } };
  }

  /** Writes the session's transcript: two turns, the first ending on an attachment. */
  async function writeTranscript(sessionId: string) {
    const keys = ["p1", "r1", "t1", "p2", "r2"];
    ids = Object.fromEntries(keys.map((key) => [key, randomUUID()]));
    const answer = (id: string, text: string) => ({
      id,
      role: "assistant",
      model: "claude-sonnet-4-6",
      content: [{ type: "text", text }],
    });
    const records = [
      { type: "user", message: { role: "user", content: "First prompt" } },
      { type: "assistant", message: answer("msg_one", "First answer") },
      { type: "attachment", attachment: { type: "todo" } },
      { type: "user", message: { role: "user", content: "Second prompt" } },
      { type: "assistant", message: answer("msg_two", "Second answer") },
    ].map((record, index) => ({
      ...record,
      sessionId,
      cwd,
      isSidechain: false,
      timestamp: new Date(1_700_000_000_000 + index).toISOString(),
      uuid: ids[keys[index]],
      parentUuid: index === 0 ? null : ids[keys[index - 1]],
    }));
    const projectDir = path.join(root, "config", "projects", encodeProjectPath(cwd));
    await mkdir(projectDir, { recursive: true });
    transcript = path.join(projectDir, `${sessionId}.jsonl`);
    await writeFile(transcript, records.map((record) => JSON.stringify(record)).join("\n") + "\n");
  }

  async function newSession() {
    const { sessionId } = await agent.newSession({ cwd, mcpServers: [] });
    await writeTranscript(sessionId);
    return sessionId;
  }

  function newAgent(updates: any[] = []) {
    return new ClaudeAcpAgent(
      {
        sessionUpdate: async (update: any) => {
          updates.push(update);
        },
      } as unknown as AcpClient,
      { log() {}, error() {}, warn() {} },
    );
  }

  beforeEach(async () => {
    root = await realpath(await mkdtemp(path.join(tmpdir(), "session-rewind-agent-")));
    cwd = path.join(root, "project");
    await mkdir(cwd);
    vi.stubEnv("CLAUDE_CONFIG_DIR", path.join(root, "config"));
    vi.stubEnv("ANTHROPIC_MODEL", undefined);
    queries = [];
    initializationGate = undefined;
    refuseCut = false;
    answerRewind = rewindInPlace;
    sdk.query.mockImplementation((request) => {
      const query = createQuery(request);
      queries.push(query);
      return query;
    });
    agent = newAgent();
    await initializeClient(agent, {});
  });

  afterEach(async () => {
    await agent.dispose();
    vi.unstubAllEnvs();
    vi.clearAllMocks();
    await rm(root, { recursive: true, force: true });
  });

  it("is advertised in initialize", async () => {
    const response = await agent.initialize({ protocolVersion: 1, clientCapabilities: {} });
    expect(response.agentCapabilities?._meta?.["sh.rexa/sessionRewind"]).toEqual({});
  });

  it("rewinds the live query in place", async () => {
    const sessionId = await newSession();
    const before = agent.sessions[sessionId];

    await expect(agent.rewindSession({ sessionId, toMessageId: "msg_one" })).resolves.toEqual({
      lastMessageId: "msg_one",
    });

    expect(queries).toHaveLength(1);
    expect(queries[0].request).toHaveBeenCalledExactlyOnceWith({
      subtype: "rewind_conversation",
      target_message_uuid: ids.p2,
      last_seen_user_message_uuid: ids.p2,
    });
    expect(queries[0].close).not.toHaveBeenCalled();
    expect(agent.sessions[sessionId]).toBe(before);
    expect(Object.fromEntries(before.messageIdToUuid)).toEqual({
      [ids.p1]: ids.p1,
      msg_one: ids.r1,
    });
  });

  it("replays only the kept history on a load before the next prompt", async () => {
    const sessionId = await newSession();
    await agent.rewindSession({ sessionId, toMessageId: "msg_one" });
    const updates: any[] = [];
    const loader = newAgent(updates);
    await initializeClient(loader, {});

    await loader.loadSession({ sessionId, cwd, mcpServers: [] });

    const texts = updates
      .map(({ update }) => update)
      .filter((update) => /^(user|agent)_message_chunk$/.test(update.sessionUpdate))
      .map((update) => update.content.text);
    expect(texts).toEqual(["First prompt", "First answer"]);
    await loader.dispose();
  });

  it.each([
    {
      name: "refuses it",
      answer: async () => ({ response: { rewound: false, reason: "stale_target" } }),
    },
    {
      name: "does not know it",
      answer: async () => {
        throw new Error("Unsupported control request subtype: rewind_conversation");
      },
    },
  ])(
    "restarts the query at the cut, with the session's choices, when Claude Code $name",
    async ({ answer }) => {
      answerRewind = answer;
      const sessionId = await newSession();
      await agent.setSessionConfigOption({ sessionId, configId: "model", value: "opus" });
      await agent.setSessionMode({ sessionId, modeId: "acceptEdits" });

      await expect(agent.rewindSession({ sessionId, toMessageId: "msg_one" })).resolves.toEqual({
        lastMessageId: "msg_one",
      });

      expect(queries).toHaveLength(2);
      expect(queries[0].close).toHaveBeenCalledOnce();
      expect(queries[1].options).toMatchObject({
        resume: sessionId,
        resumeSessionAt: ids.t1,
        resumeDropsTurn: ids.p2,
        permissionMode: "acceptEdits",
      });
      expect(queries[1].state.model).toBe("opus");
      expect(Object.fromEntries(agent.sessions[sessionId].messageIdToUuid)).toEqual({
        [ids.p1]: ids.p1,
        msg_one: ids.r1,
      });
    },
  );

  it.each([
    { name: "the rewind", restart: false },
    { name: "the restart", restart: true },
  ])("reads the history $name kept until the next prompt", async ({ restart }) => {
    if (restart) answerRewind = async () => ({ response: { rewound: false, reason: "expired" } });
    const sessionId = await newSession();
    await agent.rewindSession({ sessionId, toMessageId: "msg_one" });

    await expect(agent.rewindSession({ sessionId, toMessageId: "msg_two" })).rejects.toMatchObject({
      code: -32002,
      data: { reason: "unknown_message_id" },
    });
  });

  it("changes nothing when no message follows", async () => {
    const sessionId = await newSession();

    await expect(agent.rewindSession({ sessionId, toMessageId: "msg_two" })).resolves.toEqual({
      lastMessageId: "msg_two",
    });

    expect(queries).toHaveLength(1);
    expect(queries[0].request).not.toHaveBeenCalled();
  });

  it.each([
    { name: "a turn is queued", queued: true, answer: rewindInPlace },
    {
      name: "Claude Code reports a running turn",
      queued: false,
      answer: async () => ({ response: { rewound: false, reason: "turn_running" } }),
    },
  ])("refuses while $name", async ({ queued, answer }) => {
    answerRewind = answer;
    const sessionId = await newSession();
    if (queued) agent.sessions[sessionId].turnQueue = [{ settled: false } as any];

    await expect(agent.rewindSession({ sessionId, toMessageId: "msg_one" })).rejects.toMatchObject({
      data: { reason: "turn_running" },
    });
    expect(queries).toHaveLength(1);
  });

  it("refuses an unknown message id", async () => {
    const sessionId = await newSession();

    await expect(
      agent.rewindSession({ sessionId, toMessageId: "msg_missing" }),
    ).rejects.toMatchObject({ code: -32002, data: { reason: "unknown_message_id" } });
    expect(queries[0].request).not.toHaveBeenCalled();
  });

  it("refuses a session this connection does not hold", async () => {
    await expect(
      agent.rewindSession({ sessionId: randomUUID(), toMessageId: "msg_one" }),
    ).rejects.toMatchObject({ code: -32002, data: { sessionId: expect.any(String) } });
  });

  it("resumes the whole conversation when Claude Code refuses the restart's cut", async () => {
    answerRewind = async () => ({ response: { rewound: false, reason: "persist_failed" } });
    refuseCut = true;
    const sessionId = await newSession();

    await expect(agent.rewindSession({ sessionId, toMessageId: "msg_one" })).rejects.toMatchObject({
      data: { reason: "rewind_refused" },
    });

    expect(queries).toHaveLength(3);
    expect(queries[2].options.resume).toBe(sessionId);
    expect(queries[2].options.resumeSessionAt).toBeUndefined();
    expect(agent.sessions[sessionId].query).toBe(queries[2]);
  });

  it("runs a prompt sent during the rewind after it", async () => {
    const sessionId = await newSession();
    const gate = deferred();
    answerRewind = async (control) => {
      await gate.promise;
      return rewindInPlace(control);
    };

    const rewind = agent.rewindSession({ sessionId, toMessageId: "msg_one" });
    const prompt = agent.prompt({ sessionId, prompt: [{ type: "text", text: "Again" }] });
    await vi.waitFor(() => expect(queries[0].request).toHaveBeenCalled());
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(queries[0].prompts).toEqual([]);
    gate.resolve();
    await rewind;

    await expect(prompt).resolves.toMatchObject({ stopReason: "end_turn" });
    expect(queries[0].prompts).toEqual(["Again"]);
  });
});
