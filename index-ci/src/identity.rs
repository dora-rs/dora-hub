//! Namespace-identity binding (spec §7.4): a *newly-claimed* namespace must
//! match the claiming PR author's GitHub identity — their own login, or an
//! organization they belong to.
//!
//! GitHub only exposes *public* org membership, and most memberships are
//! private, so "not a public member" is ambiguous (impersonation **or** a
//! legitimate private member). We therefore only **reject** the one case we can
//! confirm is wrong — a namespace that is another real user's login — and route
//! every ambiguous case (org with non-public membership, unknown account, API
//! unreachable) to **human review** rather than failing. New namespaces already
//! get a mandatory human (`decide` HOLDs them); this adds an auto-verify signal
//! and a hard stop for personal-namespace impersonation (#2201: "CI rejects a
//! namespace-identity mismatch").
//!
//! Lookups shell out to `curl` against the public GitHub API (mirroring how
//! `git.rs` shells out to `git`), keeping the crate's offline-by-default
//! dependency set. `GITHUB_TOKEN`/`GH_TOKEN`, if set, only lifts the rate limit.

use std::process::Command;

use eyre::Context;

use crate::git;

/// Whether `author` is a *public* member of the org (from `public_members`).
#[derive(Debug, PartialEq, Eq)]
enum Membership {
    Member,
    NotMember,
    Unverifiable,
}

/// What the claimed namespace resolves to on GitHub.
#[derive(Debug, PartialEq, Eq)]
enum Account {
    /// An organization, plus whether the author is a public member.
    Org(Membership),
    /// A real user account (and — checked by the caller — not the author).
    User,
    /// No such user or org.
    None,
    /// The API could not be reached / gave an unexpected status.
    Unverifiable,
}

enum Verdict {
    Verified(String),
    Review(String),
    Reject(String),
}

/// GitHub logins are case-insensitive, so a personal namespace matches its
/// owner's login regardless of case.
fn personal_match(ns: &str, author: &str) -> bool {
    ns.eq_ignore_ascii_case(author)
}

/// Map a `public_members` HTTP status: 204 = public member, 404 = not public
/// (private membership or non-member), anything else = can't verify.
fn membership_from_status(status: Option<u16>) -> Membership {
    match status {
        Some(204) => Membership::Member,
        Some(404) => Membership::NotMember,
        _ => Membership::Unverifiable,
    }
}

/// The verdict for one non-login-matching namespace, given what it resolves to.
/// Pure — the IO lives in `classify` / `http_status`.
fn verdict(ns: &str, author: &str, account: Account) -> Verdict {
    match account {
        Account::Org(Membership::Member) => {
            Verdict::Verified(format!("`{author}` is a public member of org `{ns}`"))
        }
        Account::Org(Membership::NotMember) => Verdict::Review(format!(
            "`{author}` is not a *public* member of org `{ns}` (membership may be \
             private) — confirm the author belongs to it"
        )),
        Account::Org(Membership::Unverifiable) => Verdict::Review(format!(
            "could not verify org `{ns}` membership for `{author}` — review by hand"
        )),
        // a real user account that is not the author (login match was checked first)
        Account::User => Verdict::Reject(format!(
            "namespace `{ns}` is another user's GitHub login, not author `{author}` \
             — impersonation (§7.4)"
        )),
        Account::None => Verdict::Review(format!(
            "namespace `{ns}` matches no GitHub user or org — confirm author \
             `{author}` owns it"
        )),
        Account::Unverifiable => Verdict::Review(format!(
            "could not reach GitHub to verify namespace `{ns}` — re-run or review by hand"
        )),
    }
}

/// Resolve what a claimed namespace is on GitHub (org vs user vs nothing) and,
/// for an org, whether `author` is a public member.
fn classify(ns: &str, author: &str) -> eyre::Result<Account> {
    // guarded before building any URL (both are validated key-parts/logins, but
    // re-check defensively so a bad value can never reach the API path)
    if !crate::is_valid_key_part(ns) || !crate::is_valid_login(author) {
        return Ok(Account::None);
    }
    match http_status(&format!("/orgs/{ns}"))? {
        Some(s) if (200..300).contains(&s) => {
            let m = membership_from_status(http_status(&format!(
                "/orgs/{ns}/public_members/{author}"
            ))?);
            Ok(Account::Org(m))
        }
        Some(404) => match http_status(&format!("/users/{ns}"))? {
            Some(s) if (200..300).contains(&s) => Ok(Account::User),
            Some(404) => Ok(Account::None),
            _ => Ok(Account::Unverifiable),
        },
        _ => Ok(Account::Unverifiable),
    }
}

/// `GET https://api.github.com{path}` returning just the HTTP status, via `curl`.
fn http_status(path: &str) -> eyre::Result<Option<u16>> {
    let url = format!("https://api.github.com{path}");
    let mut cmd = Command::new("curl");
    cmd.args([
        "-s",
        "-o",
        "/dev/null",
        "-w",
        "%{http_code}",
        "-H",
        "Accept: application/vnd.github+json",
        "-H",
        "User-Agent: dora-index-ci",
    ]);
    if let Some(token) = token() {
        cmd.args(["-H", &format!("Authorization: Bearer {token}")]);
    }
    cmd.arg(&url);
    let out = cmd
        .output()
        .context("failed to run `curl` for the GitHub identity check")?;
    Ok(String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse::<u16>()
        .ok())
}

fn token() -> Option<String> {
    ["GITHUB_TOKEN", "GH_TOKEN"]
        .iter()
        .find_map(|k| std::env::var(k).ok())
        .filter(|t| !t.is_empty())
}

/// Verify the identity binding of every namespace newly claimed in `base..HEAD`.
/// Exits non-zero only on a confirmed impersonation; ambiguous claims are
/// flagged for human review (the new-namespace human gate in `decide` still
/// applies).
pub fn run(author: &str, base: &str) -> eyre::Result<i32> {
    let base_ns = git::namespaces_at(base)?;
    let head_ns = git::namespaces_at("HEAD")?;
    let new: Vec<String> = head_ns.difference(&base_ns).cloned().collect();

    if new.is_empty() {
        println!("identity: OK (no new namespace claims)");
        return Ok(0);
    }

    let mut verified = 0usize;
    let mut review: Vec<String> = Vec::new();
    let mut reject: Vec<String> = Vec::new();
    for ns in &new {
        if personal_match(ns, author) {
            println!("identity: `{ns}` matches author login `{author}`");
            verified += 1;
            continue;
        }
        match verdict(ns, author, classify(ns, author)?) {
            Verdict::Verified(msg) => {
                println!("identity: {msg}");
                verified += 1;
            }
            Verdict::Review(msg) => review.push(msg),
            Verdict::Reject(msg) => reject.push(msg),
        }
    }

    for r in &review {
        println!("::warning::{r}");
    }
    for r in &reject {
        println!("::error::{r}");
    }
    if !reject.is_empty() {
        println!(
            "identity: {} namespace claim(s) REJECTED as impersonation (§7.4)",
            reject.len()
        );
        return Ok(1);
    }
    if review.is_empty() {
        println!("identity: OK ({verified} new namespace claim(s) verified)");
    } else {
        println!(
            "identity: {verified} verified, {} flagged for human review (not a failure)",
            review.len()
        );
    }
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn personal_match_is_case_insensitive() {
        assert!(personal_match("octocat", "octocat"));
        assert!(personal_match("OctoCat", "octocat"));
        assert!(!personal_match("dora-rs", "octocat"));
    }

    #[test]
    fn membership_status_mapping() {
        assert_eq!(membership_from_status(Some(204)), Membership::Member);
        assert_eq!(membership_from_status(Some(404)), Membership::NotMember);
        assert_eq!(membership_from_status(Some(403)), Membership::Unverifiable);
        assert_eq!(membership_from_status(Some(0)), Membership::Unverifiable);
        assert_eq!(membership_from_status(None), Membership::Unverifiable);
    }

    #[test]
    fn only_a_public_org_member_is_verified() {
        assert!(matches!(
            verdict("acme", "bob", Account::Org(Membership::Member)),
            Verdict::Verified(_)
        ));
    }

    #[test]
    fn private_or_unverifiable_org_membership_routes_to_review_not_reject() {
        // the false-reject trap: a legit private member must NOT be rejected
        assert!(matches!(
            verdict("acme", "bob", Account::Org(Membership::NotMember)),
            Verdict::Review(_)
        ));
        assert!(matches!(
            verdict("acme", "bob", Account::Org(Membership::Unverifiable)),
            Verdict::Review(_)
        ));
    }

    #[test]
    fn claiming_another_users_login_is_rejected() {
        assert!(matches!(
            verdict("torvalds", "bob", Account::User),
            Verdict::Reject(_)
        ));
    }

    #[test]
    fn unknown_or_unverifiable_account_routes_to_review() {
        assert!(matches!(
            verdict("acme", "bob", Account::None),
            Verdict::Review(_)
        ));
        assert!(matches!(
            verdict("acme", "bob", Account::Unverifiable),
            Verdict::Review(_)
        ));
    }
}
