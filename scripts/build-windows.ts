#!/usr/bin/env bun
/**
 * Build Windows de Parler : vérifie que la version est identique partout,
 * lance `tauri build`, puis copie l'installateur sous un nom stable :
 *   src-tauri/target/release/bundle/Parler-Setup-<version>.exe
 *
 * Usage : bun run build:windows
 */
import { copyFileSync, existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

const root = join(import.meta.dir, "..");

function readJsonVersion(path: string): string {
  return JSON.parse(readFileSync(join(root, path), "utf-8")).version;
}

function readCargoVersion(): string {
  const cargo = readFileSync(join(root, "src-tauri", "Cargo.toml"), "utf-8");
  const match = cargo.match(/^\s*version\s*=\s*"([^"]+)"/m);
  if (!match) throw new Error("Version introuvable dans src-tauri/Cargo.toml");
  return match[1];
}

const versions = {
  "package.json": readJsonVersion("package.json"),
  "src-tauri/tauri.conf.json": readJsonVersion("src-tauri/tauri.conf.json"),
  "src-tauri/Cargo.toml": readCargoVersion(),
};

const distinct = new Set(Object.values(versions));
if (distinct.size !== 1) {
  console.error("Versions incohérentes :", versions);
  process.exit(1);
}
const version = versions["package.json"];

const changelog = readFileSync(join(root, "CHANGELOG.md"), "utf-8");
if (!changelog.includes(`## [${version}]`)) {
  console.error(`CHANGELOG.md n'a pas d'entrée pour la version ${version}.`);
  process.exit(1);
}

console.log(`Build de Parler ${version}…`);
const build = Bun.spawnSync(["bun", "run", "tauri", "build"], {
  cwd: root,
  stdout: "inherit",
  stderr: "inherit",
});
if (build.exitCode !== 0) {
  process.exit(build.exitCode ?? 1);
}

const bundleDir = join(root, "src-tauri", "target", "release", "bundle");
const nsisInstaller = join(
  bundleDir,
  "nsis",
  `Parler_${version}_x64-setup.exe`,
);
if (!existsSync(nsisInstaller)) {
  console.error(`Installateur introuvable : ${nsisInstaller}`);
  process.exit(1);
}
const target = join(bundleDir, `Parler-Setup-${version}.exe`);
copyFileSync(nsisInstaller, target);
console.log(`Livrable : ${target}`);
