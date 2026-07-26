export function parseOptionalRefineIterations(raw: string): number | undefined {
  const value = raw.trim();
  if (value === "") return undefined;
  if (!/^\d+$/.test(value)) return undefined;
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) ? parsed : undefined;
}

export function isValidOptionalRefineIterations(raw: string): boolean {
  return raw.trim() === "" || parseOptionalRefineIterations(raw) !== undefined;
}
