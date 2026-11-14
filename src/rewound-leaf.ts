import { open } from "node:fs/promises";

/** The size of one backward read of a transcript. */
const TAIL_CHUNK_BYTES = 64 * 1024;

type TranscriptRecord = {
  type?: unknown;
  isSidechain?: unknown;
  leafUuid?: unknown;
  explicit?: unknown;
};

/**
 * Finds the leaf that a rewind of Claude Code left in a transcript, reading
 * its records from the newest.
 *
 * Claude Code's `rewind_conversation` (and the terminal's `/rewind`) keeps the
 * discarded messages in the transcript and appends a `last-prompt` checkpoint
 * `{ leafUuid, explicit: true, rewound: true }` naming the last message kept.
 * Its resume continues from that leaf, while the SDK's `getSessionMessages`
 * still ends at the newest message, the discarded one. A later checkpoint of
 * the same leaf keeps it explicit, as Claude Code reads it; a message written
 * after the checkpoint (the next prompt) means the conversation went on from
 * the leaf, and the newest message is current again.
 */
export class RewoundLeafScan {
  private latest: string | undefined;
  /** The leaf, once the scan found the explicit checkpoint. */
  leaf: string | undefined;

  /** Visits the previous record. Returns true when the scan has its answer. */
  visit(record: TranscriptRecord): boolean {
    if ((record.type === "user" || record.type === "assistant") && record.isSidechain !== true) {
      return true;
    }
    if (record.type !== "last-prompt") return false;
    if (typeof record.leafUuid !== "string") return true;
    this.latest ??= record.leafUuid;
    if (record.leafUuid !== this.latest) return true;
    if (record.explicit !== true) return false;
    this.leaf = this.latest;
    return true;
  }
}

/** The leaf a rewind left in the JSONL transcript at `filePath`, if any (see {@link RewoundLeafScan}). */
export async function rewoundLeafOfTranscript(filePath: string): Promise<string | undefined> {
  const scan = new RewoundLeafScan();
  const handle = await open(filePath, "r");
  try {
    let end = (await handle.stat()).size;
    // The bytes of the line whose start is not read yet, in file order.
    let pending: Buffer[] = [];
    while (end > 0) {
      const start = Math.max(0, end - TAIL_CHUNK_BYTES);
      let chunk = Buffer.alloc(end - start);
      await handle.read(chunk, 0, chunk.length, start);
      for (let at = chunk.lastIndexOf(0x0a); at >= 0; at = chunk.lastIndexOf(0x0a)) {
        if (visitLine(scan, Buffer.concat([chunk.subarray(at + 1), ...pending]))) return scan.leaf;
        pending = [];
        chunk = chunk.subarray(0, at);
      }
      pending.unshift(chunk);
      end = start;
    }
    visitLine(scan, Buffer.concat(pending));
    return scan.leaf;
  } finally {
    await handle.close();
  }
}

/** `messages` up to and including `leaf`, or all of them when `leaf` is not among them. */
export function endAtLeaf<M extends { uuid?: string | null }>(messages: M[], leaf: string): M[] {
  const end = messages.findIndex((message) => message.uuid === leaf);
  return end < 0 ? messages : messages.slice(0, end + 1);
}

/** Parses only the lines whose type the scan reads: most lines are large and of no interest. */
function visitLine(scan: RewoundLeafScan, line: Buffer): boolean {
  if (
    line.indexOf('"last-prompt"') < 0 &&
    line.indexOf('"user"') < 0 &&
    line.indexOf('"assistant"') < 0
  ) {
    return false;
  }
  let record: unknown;
  try {
    record = JSON.parse(line.toString("utf8"));
  } catch {
    return false;
  }
  return record !== null && typeof record === "object" && scan.visit(record as TranscriptRecord);
}
