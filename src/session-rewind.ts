import { RequestError } from "@agentclientprotocol/sdk/unstable";
import {
  getSessionMessages,
  type SessionMessage,
  type SessionStoreEntry,
} from "@anthropic-ai/claude-agent-sdk";
import { isUnknownSessionError } from "./fork-session.js";
import { loadFullSessionHistory } from "./resumed-session.js";
import { endAtLeaf, RewoundLeafScan } from "./rewound-leaf.js";
import { messageIdForGrouping } from "./sdk-to-acp.js";
import { raceTimeoutAndAbort } from "./utils.js";

/**
 * Rexa's vendor method that rewinds a session in place: `{ sessionId,
 * toMessageId }` keeps the history up to and including `toMessageId` and
 * discards the rest, under the same session id. It is the `session/rewind` of
 * ACP RFD #1321 under Rexa's name, so moving to the standard method is a
 * rename. The contract is Rexa's `docs/acp-adapter-forks.md`.
 */
export const SESSION_REWIND_METHOD = "_sh.rexa/session/rewind";

/** The `agentCapabilities._meta` key that advertises {@link SESSION_REWIND_METHOD}. */
const SESSION_REWIND_CAPABILITY = "sh.rexa/sessionRewind";

/** The CLI's message when `resumeDropsTurn` refuses a resume (see the SDK's `Options.resumeDropsTurn`). */
const DROPS_TURN_REFUSAL = "Resume rejected by --resume-drops-turn:";

/** The text of the user record Claude Code writes after an interrupted turn. */
const INTERRUPT_MARKER = /^\[Request interrupted by user/;

/** Claude Code's refusals of `rewind_conversation` that mean it is busy: a
 *  restart would end that work, so the client is told to wait instead. */
const BUSY_REFUSALS = new Set(["turn_running", "commands_queued", "prompt_pending"]);

/** How long `rewind_conversation` may take before the restart takes over. */
const IN_PLACE_TIMEOUT_MS = 30_000;

export type RewindSessionRequest = {
  sessionId: string;
  /** A `messageId` the agent sent for this session on a message chunk, live or replayed. */
  toMessageId: string;
};

export type RewindSessionResponse = {
  /** The last message kept: `toMessageId`. */
  lastMessageId: string;
};

/** Where the restarted query resumes, as the SDK options of the same names. */
export type RewindCut = {
  /** The last transcript record of what is kept. */
  resumeSessionAt: string;
  /** The prompt of the one turn the cut discards, for the CLI to check the cut against. */
  resumeDropsTurn?: string;
};

type RewindPlan = {
  cut: RewindCut;
  /** The uuid of the first message discarded. */
  firstDiscarded: string;
  /** Whether that message is an assistant message (a cut inside a turn). */
  firstDiscardedIsAssistant: boolean;
  /** The uuid of the last prompt of the history read, discarded or not. */
  lastPrompt: string | undefined;
  /** The ACP message ids of the messages kept, with the uuid each resolves to. */
  keptMessageIds: Map<string, string>;
};

/** The session record fields a rewind reads. */
type RewindableSession = {
  messageIdToUuid: Map<string, string>;
  query: unknown;
  queryClosed?: boolean;
};

/** The SDK's `Query.request`, which sends a control request and resolves
 *  with its `control_response`. The SDK does not type it. */
type ControlRequester = {
  request?: (request: Record<string, unknown>) => Promise<unknown>;
};

type InPlaceOutcome =
  { kind: "rewound" } | { kind: "busy"; reason: string } | { kind: "fallback"; reason: string };

/** What the agent lends {@link SessionRewinds} for one rewind. */
export type SessionRewindHost<S> = {
  /** Resolves when no provider update is pending. */
  ready(): Promise<void>;
  /** The live record of the session, if this connection holds it. */
  session(sessionId: string): S | undefined;
  /** Whether Claude is working in the session: a turn, or a cycle Claude Code started. */
  isWorking(sessionId: string, session: S): boolean;
  /** Runs `change` after the configuration changes queued for the session. */
  serialize<T>(sessionId: string, change: () => Promise<T>): Promise<T>;
  /** Closes the session's query and resumes the conversation in a new one, at
   *  `cut` when given, with the session's choices. Returns the new record,
   *  or `undefined` when the session closed meanwhile. */
  restart(sessionId: string, session: S, cut?: RewindCut): Promise<S | undefined>;
  logger: { log: (...args: unknown[]) => void; error: (...args: unknown[]) => void };
};

/**
 * The rewinds of one connection's sessions.
 *
 * A rewind sends Claude Code's `rewind_conversation` control request on the
 * session's live query: Claude Code drops the messages from the first one
 * discarded on, in memory, and appends a `last-prompt` checkpoint naming the
 * last message kept, which its resume follows (see `RewoundLeafScan`). The
 * query, its process, MCP servers and choices stay as they are. The SDK does
 * not type that request; the CLI answers it since 2.1.325 at the latest.
 *
 * When Claude Code refuses it (other than as busy), does not know it, or the
 * query has ended, the rewind restarts the query with `resume` and
 * `resumeSessionAt` instead. That cut lives only in the new query: it writes
 * nothing to the transcript, so a query started again before the next prompt
 * resumes the whole conversation, and until that prompt a later rewind of the
 * same query reads the history up to the first discarded message.
 */
export class SessionRewinds<S extends RewindableSession> {
  /** The first message the query of a record left out, for a query that
   *  resumed at a cut. Keyed by the record: a query started without the cut
   *  replaces the record, and the entry goes with it. */
  private readonly firstDiscarded = new WeakMap<S, string>();
  /** The rewind each session is going through. */
  private readonly running = new Map<string, Promise<void>>();

  /** The rewind `sessionId` is going through, if any: a prompt waits for it,
   *  so it runs on the query that the rewind starts. */
  inFlight(sessionId: string): Promise<void> | undefined {
    return this.running.get(sessionId);
  }

  rewind(params: RewindSessionRequest, host: SessionRewindHost<S>): Promise<RewindSessionResponse> {
    // Registered before the first await, so a prompt that arrives later waits.
    const rewind = host
      .ready()
      .then(() => host.serialize(params.sessionId, () => this.rewindIdle(params, host)));
    const settled = rewind.then(
      () => {},
      () => {},
    );
    this.running.set(params.sessionId, settled);
    void settled.then(() => {
      if (this.running.get(params.sessionId) === settled) this.running.delete(params.sessionId);
    });
    return rewind;
  }

  private async rewindIdle(
    { sessionId, toMessageId }: RewindSessionRequest,
    host: SessionRewindHost<S>,
  ): Promise<RewindSessionResponse> {
    const session = idleSession(sessionId, host);
    const plan = await planRewind(sessionId, toMessageId, this.firstDiscarded.get(session));
    if (!plan) return { lastMessageId: toMessageId };
    // The transcript read awaited: the client may have closed the session, or
    // Claude Code may have started a cycle of its own.
    idleSession(sessionId, host, session);
    const startedAt = performance.now();
    const inPlace = await rewindInPlace(session, plan);
    const took = `${Math.round(performance.now() - startedAt)}ms`;
    if (inPlace.kind === "rewound") {
      host.logger.log(
        `Session ${sessionId}: rewound in place to ${toMessageId} without ${plan.firstDiscarded} (${took})`,
      );
      session.messageIdToUuid.clear();
      for (const [messageId, uuid] of plan.keptMessageIds) {
        session.messageIdToUuid.set(messageId, uuid);
      }
      return { lastMessageId: toMessageId };
    }
    if (inPlace.kind === "busy") {
      throw turnRunningError(sessionId);
    }
    host.logger.log(
      `Session ${sessionId}: no in-place rewind (${inPlace.reason}, ${took}); restarting the query at ${plan.cut.resumeSessionAt}` +
        (plan.cut.resumeDropsTurn ? ` without turn ${plan.cut.resumeDropsTurn}` : ""),
    );
    let rewound: S | undefined;
    try {
      rewound = await host.restart(sessionId, session, plan.cut);
    } catch (error) {
      throw await recoverFromFailedRewind(sessionId, session, error, host);
    }
    if (!rewound) throw RequestError.sessionNotFound(sessionId);
    this.firstDiscarded.set(rewound, plan.firstDiscarded);
    for (const [messageId, uuid] of plan.keptMessageIds) {
      rewound.messageIdToUuid.set(messageId, uuid);
    }
    return { lastMessageId: toMessageId };
  }
}

/** `agentCapabilities._meta` entries that advertise the rewind. */
export function sessionRewindCapability(): Record<string, object> {
  return { [SESSION_REWIND_CAPABILITY]: {} };
}

export function parseRewindSessionRequest(value: unknown): RewindSessionRequest {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw RequestError.invalidParams(undefined, "rewind params must be an object");
  }
  const { sessionId, toMessageId } = value as Record<string, unknown>;
  if (typeof sessionId !== "string" || sessionId.length === 0) {
    throw RequestError.invalidParams(undefined, "rewind params require a non-empty sessionId");
  }
  if (typeof toMessageId !== "string" || toMessageId.length === 0) {
    throw RequestError.invalidParams(undefined, "rewind params require a non-empty toMessageId");
  }
  return { sessionId, toMessageId };
}

/**
 * Where to resume the session `sessionId` so that its history ends with
 * `toMessageId`, or `undefined` when nothing follows that message.
 *
 * The cut is the parent record of the first message after `toMessageId`:
 * the last record of what is kept, so the tool results, attachments and
 * system records that belong with `toMessageId` stay. A message after it is a
 * prompt, or an assistant message with another id. An interrupt marker is
 * not a message: it belongs to the interrupted turn. `resumeDropsTurn` is
 * declared only when the cut discards exactly one turn, from its prompt on.
 *
 * `firstDiscarded` is the first message an earlier rewind of the live query
 * left out: the history read ends before it.
 */
export async function planRewind(
  sessionId: string,
  toMessageId: string,
  firstDiscarded?: string,
): Promise<RewindPlan | undefined> {
  const records = await readTranscript(sessionId);
  const messages = await activeMessages(sessionId, records, firstDiscarded);
  const target = messages.findLastIndex((message) => messageIdForGrouping(message) === toMessageId);
  if (target < 0) throw unknownMessageIdError(sessionId, toMessageId);
  const next = messages.findIndex(
    (message, index) => index > target && startsMessage(message, toMessageId),
  );
  if (next < 0) return undefined;
  const discarded = messages[next];
  const dropsOneTurn = isPrompt(discarded) && !messages.slice(next + 1).some(isPrompt);
  return {
    cut: {
      resumeSessionAt: cutBefore(recordsByUuid(records), messages, next),
      ...(dropsOneTurn ? { resumeDropsTurn: discarded.uuid } : {}),
    },
    firstDiscarded: discarded.uuid,
    firstDiscardedIsAssistant: discarded.type === "assistant",
    lastPrompt: messages.findLast(isPrompt)?.uuid,
    keptMessageIds: messageIds(messages.slice(0, next)),
  };
}

/**
 * Asks Claude Code to drop `plan.firstDiscarded` and everything after it from
 * the live query. `last_seen_user_message_uuid` names the last prompt the
 * client saw, without which Claude Code refuses a target that has later
 * turns (`stale_target`). An assistant target is allowed only right after a
 * tool result; elsewhere Claude Code refuses it, and the restart cuts there.
 */
async function rewindInPlace(
  session: RewindableSession,
  plan: RewindPlan,
): Promise<InPlaceOutcome> {
  if (session.queryClosed) return { kind: "fallback", reason: "the query has ended" };
  const request = (session.query as ControlRequester | null | undefined)?.request;
  if (typeof request !== "function") {
    return { kind: "fallback", reason: "the SDK's Query sends no control requests" };
  }
  const answer = await raceTimeoutAndAbort(
    request
      .call(session.query, {
        subtype: "rewind_conversation",
        target_message_uuid: plan.firstDiscarded,
        ...(plan.lastPrompt ? { last_seen_user_message_uuid: plan.lastPrompt } : {}),
        ...(plan.firstDiscardedIsAssistant ? { allow_assistant_target: true } : {}),
      })
      .then(
        (value) => ({ value }),
        (error: unknown) => ({ error }),
      ),
    IN_PLACE_TIMEOUT_MS,
    new AbortController().signal,
  );
  if (answer.type !== "done") {
    return { kind: "fallback", reason: `no answer within ${IN_PLACE_TIMEOUT_MS}ms` };
  }
  if ("error" in answer.value) {
    const { error } = answer.value;
    return { kind: "fallback", reason: error instanceof Error ? error.message : String(error) };
  }
  // The SDK resolves with the whole `control_response`; its `response` is the CLI's answer.
  const reply = answer.value.value as { response?: unknown } | null | undefined;
  const response = (reply?.response ?? reply) as
    { rewound?: unknown; reason?: unknown; error?: unknown } | null | undefined;
  if (response?.rewound === true) return { kind: "rewound" };
  const reason = typeof response?.reason === "string" ? response.reason : "no reason";
  if (BUSY_REFUSALS.has(reason)) return { kind: "busy", reason };
  return {
    kind: "fallback",
    reason: `Claude Code refused it: ${reason}${typeof response?.error === "string" ? ` (${response.error})` : ""}`,
  };
}

/** The live record of `sessionId`, idle; `expected` is the record the rewind started on. */
function idleSession<S>(sessionId: string, host: SessionRewindHost<S>, expected?: S): S {
  const session = host.session(sessionId);
  if (!session || (expected !== undefined && session !== expected)) {
    throw RequestError.sessionNotFound(sessionId);
  }
  if (host.isWorking(sessionId, session)) throw turnRunningError(sessionId);
  return session;
}

function turnRunningError(sessionId: string): RequestError {
  return RequestError.invalidRequest(
    { reason: "turn_running", sessionId },
    "Claude is working in this session, so it can't be rewound. Try again when the turn ends.",
  );
}

/**
 * The restart at the cut failed after it closed the session's query: resumes
 * the whole conversation, so the session stays usable, and returns the error
 * for the client. A refusal of `resumeDropsTurn` is not retried with the same
 * cut: the CLI refuses it every time.
 */
async function recoverFromFailedRewind<S>(
  sessionId: string,
  session: S,
  error: unknown,
  host: SessionRewindHost<S>,
): Promise<unknown> {
  host.logger.error(`Session ${sessionId}: the rewind failed:`, error);
  let recovered = false;
  if (host.session(sessionId) === session) {
    try {
      recovered = (await host.restart(sessionId, session)) !== undefined;
    } catch (recoveryError) {
      host.logger.error(
        `Session ${sessionId}: could not resume it after the failed rewind:`,
        recoveryError,
      );
    }
  }
  if (error instanceof RequestError) return error;
  const cause = error instanceof Error ? error.message : String(error);
  const outcome = recovered
    ? "The session keeps its whole history."
    : "The session has ended. Load it again to continue.";
  return RequestError.internalError(
    { reason: cause.includes(DROPS_TURN_REFUSAL) ? "rewind_refused" : "restart_failed", sessionId },
    `Could not rewind session ${sessionId}: ${cause}. ${outcome}`,
  );
}

/** Every record of the transcript, or none for a session Claude Code never wrote. */
async function readTranscript(sessionId: string): Promise<SessionStoreEntry[]> {
  try {
    return await loadFullSessionHistory(sessionId);
  } catch (error) {
    if (isUnknownSessionError(error, sessionId)) return [];
    throw error;
  }
}

/** The main thread's messages of the transcript's active branch, as the SDK
 *  reads them from `records`. They end at the leaf an in-place rewind left,
 *  and before `firstDiscarded` when it is on them. */
async function activeMessages(
  sessionId: string,
  records: SessionStoreEntry[],
  firstDiscarded: string | undefined,
): Promise<SessionMessage[]> {
  if (records.length === 0) return [];
  const chain = (
    await getSessionMessages(sessionId, {
      sessionStore: { append: async () => {}, load: async () => records },
    })
  ).filter((message) => message.parent_tool_use_id == null && message.parent_agent_id == null);
  const scan = new RewoundLeafScan();
  for (let index = records.length - 1; index >= 0 && !scan.visit(records[index]); index--);
  const messages = scan.leaf ? endAtLeaf(chain, scan.leaf) : chain;
  const end = firstDiscarded
    ? messages.findIndex((message) => message.uuid === firstDiscarded)
    : -1;
  return end < 0 ? messages : messages.slice(0, end);
}

/** Whether `message` comes after the message `messageId` rather than with it. */
function startsMessage(message: SessionMessage, messageId: string): boolean {
  return message.type === "assistant"
    ? messageIdForGrouping(message) !== messageId
    : isPrompt(message);
}

/** A user message that starts a turn: not tool results, and not an interrupt marker. */
function isPrompt(message: SessionMessage): boolean {
  if (message.type !== "user") return false;
  const content = (message.message as { content?: unknown } | null | undefined)?.content;
  const blocks: unknown[] =
    typeof content === "string"
      ? [{ type: "text", text: content }]
      : Array.isArray(content)
        ? content
        : [];
  if (blocks.some((block) => (block as { type?: unknown } | null)?.type === "tool_result")) {
    return false;
  }
  const text = blocks
    .map((block) => (block as { type?: unknown; text?: unknown } | null) ?? {})
    .filter((block) => block.type === "text" && typeof block.text === "string")
    .map((block) => block.text)
    .join("");
  return !INTERRUPT_MARKER.test(text.trimStart());
}

/**
 * The uuid to resume at so that `messages[next]` and everything after it is
 * discarded: the record `messages[next]` follows. It is a record the SDK's
 * message list leaves out (an attachment, a tool result's meta record, a
 * system record) when the kept turn ended with one. When the compaction of a
 * transcript relinked the messages, that record does not lead back to the
 * kept message, and the cut falls on the kept message itself.
 */
function cutBefore(
  records: Map<string, SessionStoreEntry>,
  messages: SessionMessage[],
  next: number,
): string {
  const kept = messages[next - 1].uuid;
  const parent = parentUuidOf(records.get(messages[next].uuid));
  const seen = new Set<string>();
  for (let cursor = parent; cursor && !seen.has(cursor);) {
    if (cursor === kept) return parent!;
    seen.add(cursor);
    cursor = parentUuidOf(records.get(cursor));
  }
  return kept;
}

function parentUuidOf(record: SessionStoreEntry | undefined): string | undefined {
  return typeof record?.parentUuid === "string" ? record.parentUuid : undefined;
}

/** The records by uuid; a uuid written twice keeps its last record, as Claude Code reads it. */
function recordsByUuid(records: SessionStoreEntry[]): Map<string, SessionStoreEntry> {
  return new Map(
    records.flatMap((record) => (typeof record.uuid === "string" ? [[record.uuid, record]] : [])),
  );
}

/** The ACP message id of each message, with its uuid; a split message keeps its last uuid. */
function messageIds(messages: SessionMessage[]): Map<string, string> {
  const ids = new Map<string, string>();
  for (const message of messages) {
    const messageId = messageIdForGrouping(message);
    if (messageId) ids.set(messageId, message.uuid);
  }
  return ids;
}

function unknownMessageIdError(sessionId: string, messageId: string): RequestError {
  return new RequestError(
    -32002,
    `Message ${messageId} is not in the history of session ${sessionId}. A message before the ` +
      "session's last compaction, or one an earlier rewind discarded, can't be rewound to.",
    { reason: "unknown_message_id", sessionId, messageId },
  );
}
