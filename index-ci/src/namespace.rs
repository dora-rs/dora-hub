//! Namespace governance (spec §7.4): reserved + confusable screening of newly
//! claimed namespaces. A routing signal, not a hard gate — every new namespace
//! gets at least a human reviewer; reserved claims escalate to an index admin.

use std::collections::BTreeSet;

/// The reserved list is embedded at build time so the bot runs the *base*
/// copy (not a PR-supplied one) without any file plumbing.
const RESERVED_TXT: &str = include_str!("../reserved_namespaces.txt");

#[derive(Debug, PartialEq, Eq)]
pub enum Tier {
    /// Reserved name — needs an index admin.
    Admin,
    /// New or confusable name — needs a (mandatory) human reviewer.
    Human,
}

pub fn reserved() -> BTreeSet<String> {
    RESERVED_TXT
        .lines()
        .map(|l| l.split('#').next().unwrap_or("").trim())
        .filter(|l| !l.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Collapse a namespace to its visual skeleton: drop separators and fold
/// look-alike groups, so a homoglyph swap and a structural edit can't be
/// *combined* to slip past the distance check (e.g. `d0rars` vs `dora-rs`).
/// Best-effort and intentionally over-eager — a miss only routes to review.
pub fn normalize(ns: &str) -> String {
    let mut s = ns.to_ascii_lowercase().replace(['-', '_'], "");
    for (from, to) in [("rn", "m"), ("cl", "d"), ("vv", "w"), ("nn", "m")] {
        s = s.replace(from, to);
    }
    s.chars()
        .map(|c| match c {
            '0' => 'o',
            '1' => 'l',
            '2' => 'z',
            '3' => 'e',
            '5' => 's',
            '6' | '8' => 'b',
            '9' => 'g',
            other => other,
        })
        .collect()
}

/// Levenshtein edit distance.
pub fn levenshtein(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut cur = vec![i + 1];
        for (j, &cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            cur.push((prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost));
        }
        prev = cur;
    }
    *prev.last().unwrap()
}

/// True if `a` is a distinct-but-lookalike name for `b`. Compares the visual
/// skeletons (not raw strings) so a homoglyph swap plus a single structural edit
/// still collapses to within distance 1.
pub fn is_confusable(a: &str, b: &str) -> bool {
    if a == b {
        return false;
    }
    let (na, nb) = (normalize(a), normalize(b));
    na == nb || levenshtein(&na, &nb) <= 1
}

/// Classify a *new* namespace claim. Returns `(tier, reason)`. Reserved claims
/// are `Admin`; confusable and otherwise-new claims are `Human` (confusable
/// flagged in the reason). `existing` should contain the other namespaces to
/// screen against (base namespaces + the PR's other new claims).
pub fn review_tier(
    ns: &str,
    existing: &BTreeSet<String>,
    reserved: &BTreeSet<String>,
) -> (Tier, String) {
    if reserved.contains(ns) {
        return (Tier::Admin, format!("namespace `{ns}` is reserved"));
    }
    for refn in existing.iter().chain(reserved.iter()) {
        if is_confusable(ns, refn) {
            return (
                Tier::Human,
                format!("namespace `{ns}` is confusable with `{refn}`"),
            );
        }
    }
    (Tier::Human, format!("namespace `{ns}` is newly claimed"))
}

/// Screen the namespaces a PR newly claims (base..HEAD). Emits a `::warning::`
/// per claim and returns 0 — a new namespace is a routing signal, not a CI
/// failure (exiting non-zero would force a human/admin to bypass a red check).
pub fn run(base: &str) -> eyre::Result<i32> {
    let reserved = reserved();
    let base_ns = crate::git::namespaces_at(base)?;
    let head_ns = crate::git::namespaces_at("HEAD")?;
    let new: Vec<String> = head_ns.difference(&base_ns).cloned().collect();

    for ns in &new {
        let mut existing = base_ns.clone();
        existing.extend(new.iter().filter(|n| *n != ns).cloned());
        let (tier, reason) = review_tier(ns, &existing, &reserved);
        let who = match tier {
            Tier::Admin => "an index admin",
            Tier::Human => "a human reviewer",
        };
        println!("::warning::{reason} — needs {who} before merge (§7.4); not auto-merge");
    }

    if new.is_empty() {
        println!("check_namespace: OK (no new namespaces)");
    } else {
        println!(
            "check_namespace: {} new namespace claim(s) flagged for review (not a failure)",
            new.len()
        );
    }
    Ok(0)
}
