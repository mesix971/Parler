import type { Result } from "@/bindings";

/**
 * Les commandes générées par tauri-specta ne lèvent PAS d'exception quand le
 * backend renvoie une erreur : elles résolvent `{ status: "error" }`. Un
 * simple `try/catch` autour ne voit donc rien. Ce helper transforme ce
 * résultat en exception pour que les `catch` (et les retours arrière des
 * mises à jour optimistes) fonctionnent vraiment.
 */
export class CommandError extends Error {
  constructor(public readonly detail: unknown) {
    super(typeof detail === "string" ? detail : JSON.stringify(detail));
    this.name = "CommandError";
  }
}

export function unwrapResult<T, E>(result: Result<T, E>): T {
  if (result.status === "error") {
    throw new CommandError(result.error);
  }
  return result.data;
}

/** Variante pour une promesse de commande : `await unwrap(commands.xxx())`. */
export async function unwrap<T, E>(promise: Promise<Result<T, E>>): Promise<T> {
  return unwrapResult(await promise);
}

/** Valeur de retour d'une commande, qu'elle renvoie un `Result` ou non. */
export function unwrapIfResult(value: unknown): unknown {
  if (
    value !== null &&
    typeof value === "object" &&
    "status" in value &&
    ((value as { status: unknown }).status === "ok" ||
      (value as { status: unknown }).status === "error")
  ) {
    return unwrapResult(value as Result<unknown, unknown>);
  }
  return value;
}
