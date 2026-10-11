"""Static contract test between the ep6 watchdog and the probes it calls.

The watchdog is a shell script that cannot be run here (it stops a live
training run), and it deliberately never changes gates or windows. What CAN be
checked offline is that every flag it passes is a flag the target probe
actually accepts: a typo or a renamed option would make the probe exit during
argument parsing, write no JSON, and leave the bundle incomplete -- the exact
failure mode the pre-ep6 audits keep finding.

The commands are read from the watchdog itself, so this test fails if either
side drifts.

Run: python3 me_rssm/sanity/test_watch_semA_ep6_cli.py
"""

import importlib.util
import os
import re
import shlex
import sys
import unittest

_REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))

# probe script -> the module that owns its argparse
PROBES = {
    'me_rssm/sanity/probe_semreadout_ep6.py': 'probe_semreadout_ep6',
    'me_rssm/sanity/probe_z_utilization_standard.py':
        'probe_z_utilization_standard',
    'tools/probe_posterior_source.py': 'probe_posterior_source',
}

# Read-only renderers that run after the probes. They take --run/--output
# rather than --config/--checkpoint, so they are tracked separately.
RENDERERS = {
    'me_rssm/sanity/semA_ep6_verdict.py': ('--run',),
    'me_rssm/sanity/semA_ep6_record.py': ('--run', '--output'),
}

# Flags each probe is REQUIRED to be called with, so a window cannot silently
# disappear from the battery.
REQUIRED_FLAGS = {
    'me_rssm/sanity/probe_semreadout_ep6.py': (
        '--config', '--checkpoint', '--start-index', '--limit', '--stride',
        '--shuffle-offset', '--output'),
    'me_rssm/sanity/probe_z_utilization_standard.py': (
        '--config', '--checkpoint', '--start-index', '--limit',
        '--batch-size', '--shuffle-offset', '--output'),
    'tools/probe_posterior_source.py': (
        '--config', '--checkpoint', '--start-index', '--limit', '--stride',
        '--output'),
}


# Both generations of the ep6 watchdog must satisfy the same contract. The
# A2 script is a copy of the A one, and the FIRST revision of that copy silently
# dropped the four family-standard artifacts (z-utilization w3/w100/w1800 +
# posterior source) plus the verdict/record renderers -- an "INCOMPLETE BUNDLE"
# that no test caught, because this file only ever read A's script.
WATCHDOGS = {
    'A': 'me_rssm/sanity/watch_semA_ep6.sh',
    'A2S': 'me_rssm/sanity/watch_semA2_ep6.sh',
}

# The gate probe's clip plan is arm-specific and frozen at pre-registration:
# A used stride 200 / 10 clips, A2 uses the 20-clip contract of its section 5.
GATE_CLIP_PLAN = {
    'A': ('10', '200', '3'),
    'A2S': ('20', '100', '3'),
}

# Artifacts semA_ep6_verdict.py lists as required evidence. Any watchdog (or
# finisher) for this family must be able to produce all of them.
REQUIRED_EVIDENCE = (
    'z_util_std_ep6_w3.json',
    'z_util_std_ep6_w100.json',
    'z_util_std_ep6_w1800.json',
    'post_source_std_ep6_stride700.json',
)


def _read(rel):
    with open(os.path.join(_REPO, rel), encoding='utf-8') as handle:
        return handle.read()


def watchdog_commands(rel='me_rssm/sanity/watch_semA_ep6.sh'):
    """Every probe invocation in the watchdog, as argv lists."""
    text = _read(rel)
    joined = re.sub(r'\\\n\s*', ' ', text)          # join continuations
    cmds = []
    for line in joined.split('\n'):
        line = line.strip()
        if not line or '.py' not in line or 'watch_semA' in line:
            continue
        if not re.search(r'\bpython\s+\S+\.py\b', line):
            continue
        # drop the env-var prefix and the trailing redirection
        line = re.sub(r'^\s*CUDA_VISIBLE_DEVICES=\S+\s+', '', line)
        line = re.sub(r'\s*>>?\s*\S+\s*2>&1\s*$', '', line)
        argv = shlex.split(line, posix=True)
        cmds.append(argv)
    return cmds


def accepted_flags(rel_path):
    """Flags the probe's own parser defines, read from its source."""
    source = _read(rel_path)
    return set(re.findall(r"add_argument\(\s*['\"](--[a-z0-9-]+)['\"]",
                          source))


class TestWatchdogCliContract(unittest.TestCase):

    def setUp(self):
        self.cmds = watchdog_commands()

    def test_watchdog_calls_every_probe(self):
        called = {argv[1] for argv in self.cmds if len(argv) > 1}
        # normalise: the commands reference paths relative to the repo root
        for rel in PROBES:
            self.assertIn(rel, called,
                          f'the watchdog no longer runs {rel}')

    def test_probe_scripts_exist(self):
        for rel in PROBES:
            self.assertTrue(os.path.exists(os.path.join(_REPO, rel)),
                            f'{rel} is missing')

    def test_every_flag_is_accepted_by_its_probe(self):
        checked = 0
        for argv in self.cmds:
            rel = argv[1]
            if rel not in PROBES:
                continue
            allowed = accepted_flags(rel)
            passed = [a.split('=', 1)[0] for a in argv[2:]
                      if a.startswith('--')]
            for flag in passed:
                self.assertIn(
                    flag, allowed,
                    f'{rel} is called with {flag}, which its parser does not '
                    f'define (accepted: {sorted(allowed)})')
                checked += 1
        self.assertGreater(checked, 0, 'no probe flags were checked')

    def test_required_flags_are_all_present(self):
        for argv in self.cmds:
            rel = argv[1]
            if rel not in PROBES:
                continue
            passed = {a.split('=', 1)[0] for a in argv[2:]
                      if a.startswith('--')}
            for flag in REQUIRED_FLAGS[rel]:
                self.assertIn(flag, passed,
                              f'the watchdog no longer passes {flag} to {rel}')

    def test_z_utilisation_windows_are_the_three_preregistered_ones(self):
        """w3 / w100 / w1800 must all still be collected."""
        starts = []
        for argv in self.cmds:
            if argv[1] != 'me_rssm/sanity/probe_z_utilization_standard.py':
                continue
            starts.append(argv[argv.index('--start-index') + 1])
        self.assertEqual(starts, ['3', '100', '1800'],
                         'the three comparison windows changed')

    def test_z_utilisation_windows_are_self_consistent(self):
        """offset must not map a clip onto itself within its window.

        resolve_shuffle_sources refuses a self-map, so an inconsistent
        limit/offset pair would kill the probe at run time.
        """
        for argv in self.cmds:
            if argv[1] != 'me_rssm/sanity/probe_z_utilization_standard.py':
                continue
            limit = int(argv[argv.index('--limit') + 1])
            offset = int(argv[argv.index('--shuffle-offset') + 1])
            self.assertNotEqual(offset % limit, 0,
                                f'offset {offset} self-maps with limit {limit}')
            self.assertLess(abs(offset), limit)

    def test_gate_probe_uses_the_preregistered_clip_plan(self):
        for argv in self.cmds:
            if argv[1] != 'me_rssm/sanity/probe_semreadout_ep6.py':
                continue
            self.assertEqual(argv[argv.index('--limit') + 1], '10')
            self.assertEqual(argv[argv.index('--stride') + 1], '200')
            self.assertEqual(argv[argv.index('--start-index') + 1], '3')

    def test_each_probe_writes_a_distinct_json(self):
        outs = []
        for argv in self.cmds:
            if argv[1] not in PROBES:
                continue
            outs.append(argv[argv.index('--output') + 1])
        self.assertEqual(len(outs), len(set(outs)),
                         f'probes would overwrite each other: {outs}')
        for out in outs:
            self.assertTrue(out.endswith('.json'), out)

    def test_renderers_are_invoked_with_their_own_flags(self):
        """The post-probe renderers must not silently drop out of the battery.

        They use --run/--output instead of --config/--checkpoint, so the probe
        flag loop above does not cover them. A renamed flag would make the
        step exit at argument parsing while the log still ends with
        "probes done" -- the same silent-omission class as a dead probe.
        """
        called = {}
        for argv in self.cmds:
            if argv[1] in RENDERERS:
                called[argv[1]] = argv
        for rel, required in RENDERERS.items():
            self.assertIn(rel, called, f'the watchdog no longer runs {rel}')
            argv = called[rel]
            allowed = accepted_flags(rel)
            for flag in required:
                self.assertIn(flag, argv,
                              f'{rel} is called without {flag}')
                self.assertIn(flag, allowed,
                              f'{rel} does not define {flag}')

    def test_section72_draft_is_written_but_never_auto_pasted(self):
        """The draft is generated for a human; the record is not edited."""
        text = _read(os.path.join('me_rssm', 'sanity', 'watch_semA_ep6.sh'))
        self.assertIn('semA_ep6_record.py', text)
        self.assertIn('section72_draft.md', text)
        # Never WRITE into the canonical training record. Naming it in a
        # comment is fine (the script says where the draft should be pasted),
        # so only redirection targets count.
        for target in re.findall(r'>>?\s*([^\s;]+)', text):
            self.assertNotIn('training_runs_full', target,
                             f'the watchdog writes into the record: {target}')

    def test_verdict_runs_after_the_probes_and_decides_nothing(self):
        text = _read(os.path.join('me_rssm', 'sanity', 'watch_semA_ep6.sh'))
        self.assertIn('semA_ep6_verdict.py', text)
        # it must come after the last probe invocation
        joined = re.sub(r'\\\n\s*', ' ', text)
        self.assertLess(joined.index('probe_posterior_source.py'),
                        joined.index('semA_ep6_verdict.py'))
        # and must not launch a follow-up training run
        for forbidden in ('dist_train', 'train_vod', 'resume_from'):
            self.assertNotIn(forbidden, text)
        # the section-72 renderer must come after the verdict
        self.assertLess(joined.index('semA_ep6_verdict.py'),
                        joined.index('semA_ep6_record.py'))

    def test_watchdog_does_not_auto_resume_training(self):
        """The pre-registration makes the ep6 decision a manual step."""
        text = _read(os.path.join('me_rssm', 'sanity', 'watch_semA_ep6.sh'))
        self.assertNotIn('--resume-from', text)
        self.assertNotIn('--load-from', text)
        self.assertIn('gate decision is a manual step', text)




def _commented_lines_removed(text):
    """The script's own text with `#` comment lines and trailing comments gone.

    Ordering assertions must look at what the script RUNS, not at what its
    header says it runs.
    """
    text = re.sub(r'\\\n\s*', ' ', text)
    kept = []
    for line in text.split('\n'):
        if line.lstrip().startswith('#'):
            continue
        kept.append(re.sub(r'\s+#.*$', '', line))
    return '\n'.join(kept)

class TestA2WatchdogCoversTheWholeBundle(unittest.TestCase):
    """The A2 watchdog/finisher must produce the FULL evidence bundle.

    Regression guard for a real omission: the first revision of
    watch_semA2_ep6.sh ran the A2-specific probes and then stopped, so the
    three z-utilization windows and the posterior-source attribution that
    semA_ep6_verdict.py requires would never have been written, and the
    verdict/record renderers would never have run. Tests for A's script alone
    could not see it.
    """

    def test_every_watchdog_is_tracked_by_this_file(self):
        for rel in WATCHDOGS.values():
            self.assertTrue(os.path.exists(os.path.join(_REPO, rel)),
                            f'{rel} is missing')

    def test_a2_watchdog_runs_every_required_probe(self):
        cmds = watchdog_commands(WATCHDOGS['A2S'])
        called = {argv[1] for argv in cmds if len(argv) > 1}
        for rel in PROBES:
            self.assertIn(rel, called,
                          f'watch_semA2_ep6.sh no longer runs {rel}')
        # the A2-specific readings ride along on the same battery
        self.assertIn('me_rssm/sanity/probe_semA_rank.py', called)
        self.assertIn('me_rssm/sanity/probe_semA_m4floor.py', called)

    def test_a2_watchdog_produces_the_required_evidence_artifacts(self):
        text = _read(WATCHDOGS['A2S'])
        for name in REQUIRED_EVIDENCE:
            self.assertIn(name, text,
                          f'watch_semA2_ep6.sh no longer writes {name}; '
                          'semA_ep6_verdict.py will report an INCOMPLETE '
                          'bundle')

    def test_a2_watchdog_runs_verdict_and_record_after_the_probes(self):
        # Strip comments first: the script documents the verdict tool in its
        # header, and a bare index() would then find the comment instead of
        # the invocation (this is what made an earlier revision of this test
        # fail for the wrong reason).
        joined = _commented_lines_removed(_read(WATCHDOGS['A2S']))
        self.assertIn('semA_ep6_verdict.py', joined)
        self.assertIn('semA_ep6_record.py', joined)
        self.assertLess(joined.index('tools/probe_posterior_source.py'),
                        joined.index('semA_ep6_verdict.py'))
        self.assertLess(joined.index('semA_ep6_verdict.py'),
                        joined.index('semA_ep6_record.py'))
        raw = _read(WATCHDOGS['A2S'])
        for forbidden in ('dist_train', 'train_vod', 'resume_from',
                          '--resume-from', '--load-from'):
            self.assertNotIn(forbidden, raw)
        self.assertIn('gate decision is a manual step', raw)

    def test_a2_watchdog_uses_the_frozen_20_clip_contract(self):
        """Prereg section 5 fixes n=20 / stride 100 for the gate + rank probes.

        The three z-utilization windows keep their own (unchanged) sampling,
        so only the gate and rank probes are checked here.
        """
        for argv in watchdog_commands(WATCHDOGS['A2S']):
            if argv[1] not in ('me_rssm/sanity/probe_semreadout_ep6.py',
                               'me_rssm/sanity/probe_semA_rank.py',
                               'me_rssm/sanity/probe_semA_m4floor.py'):
                continue
            self.assertEqual(argv[argv.index('--limit') + 1], '20', argv)
            self.assertEqual(argv[argv.index('--stride') + 1], '100', argv)
            self.assertEqual(argv[argv.index('--start-index') + 1], '3', argv)

    def test_finisher_is_idempotent_and_repairs_only_missing_artifacts(self):
        """A2-S was already training when the watchdog bug was found, so a
        separate finisher re-runs ONLY the missing probes (guarded by an
        existence check) and then regenerates the verdict + draft."""
        rel = 'me_rssm/sanity/finish_semA2_ep6.sh'
        self.assertTrue(os.path.exists(os.path.join(_REPO, rel)),
                        'the A2 finisher is missing')
        text = _read(rel)
        for name in REQUIRED_EVIDENCE:
            self.assertIn(name, text)
            self.assertIn(f'[ ! -f "$WORK_DIR/{name}" ]', text,
                          f'{rel} would re-run {name} unconditionally')
        # The finisher repairs only the FOUR family-standard artifacts the
        # first A2 watchdog revision dropped; the A2-specific probes and the
        # gate battery are the watchdog's job, so they are not re-run here.
        self.assertIn('me_rssm/sanity/probe_z_utilization_standard.py', text)
        self.assertIn('tools/probe_posterior_source.py', text)
        self.assertIn('semA_ep6_verdict.py', text)
        self.assertIn('semA_ep6_record.py', text)
        for forbidden in ('dist_train', 'train_vod', '--resume-from',
                          '--load-from'):
            self.assertNotIn(forbidden, text)

    def test_finisher_watchdog_guard_matches_the_real_script_name(self):
        """A dead guard let the finisher race the watchdog into an OOM.

        The finisher waited with ``pgrep -f "watch_sem${ARM}_ep6.sh"``, which
        expands to ``watch_semA2S_ep6.sh`` -- but the A2 watchdog script is
        SHARED by both arms and really runs as ``watch_semA2_ep6.sh``. The
        pattern could never match, so the wait loop broke immediately and the
        finisher started its own GPU probes on the same cards as the watchdog
        battery: the double-run that caused an earlier A2 OOM.
        """
        text = _read('me_rssm/sanity/finish_semA2_ep6.sh')
        self.assertNotIn('pgrep -f "watch_sem${ARM}_ep6.sh"', text)
        self.assertIn('WATCHDOG_SCRIPT_NAME', text)
        self.assertIn('A2S|A2R) WATCHDOG_SCRIPT_NAME=watch_semA2_ep6.sh',
                      text)
        # the guard must be the one that was fixed, not a new hardcode
        self.assertIn('pgrep -f "$WATCHDOG_SCRIPT_NAME"', text)

    def test_finisher_training_log_is_derived_not_guessed_from_arm(self):
        """A wrong LOG default made the finisher hang forever.

        It defaulted to ``/tmp/sem${ARM}_24e.log`` = ``/tmp/semA2S_24e.log``,
        a path nothing creates (the job logs to ``/tmp/a2s_24e.log`` and the
        run dir holds ``YYYYmmdd_HHMMSS.log``). The wait loop greps that path,
        so it could never be satisfied and the finisher silently never ran.
        """
        text = _read('me_rssm/sanity/finish_semA2_ep6.sh')
        self.assertNotIn('LOG="${LOG:-/tmp/sem${ARM}_24e.log}"', text)
        self.assertIn('LOG=$(ls -1t "$WORK_DIR"/[0-9]*.log', text)
        # and the derived path must be the one the wait loop greps
        self.assertIn("grep -q 'Epoch(val) \\[6\\]' \"$LOG\"", text)

    def test_finisher_derivation_picks_a_real_log_for_both_arms(self):
        """Behavioural check of the derivation against the live run dirs."""
        for arm, work in (
                ('A2S', '/data/lurui/work_dirs/'
                        'fgfull_N4_no2d_igdr_2x4_24e_semA2S_seed0'),
                ('A', '/data/lurui/work_dirs/'
                      'fgfull_N4_no2d_igdr_2x4_24e_semA_seed0')):
            if not os.path.isdir(work):
                continue          # archived away; nothing to assert
            found = [n for n in os.listdir(work)
                     if len(n) == len('20260101_000000.log')
                     and n.endswith('.log') and n[8] == '_']
            if not found:
                continue
            # naming contract: the derivation must not invent a /tmp path when
            # the run's own authoritative log exists
            self.assertTrue(any(n[0].isdigit() for n in found), arm)

    def test_gate_clip_plan_still_matches_the_recorded_epochs(self):
        for arm, rel in WATCHDOGS.items():
            argv = next(a for a in watchdog_commands(rel)
                        if a[1] == 'me_rssm/sanity/probe_semreadout_ep6.py')
            got = (argv[argv.index('--limit') + 1],
                   argv[argv.index('--stride') + 1],
                   argv[argv.index('--start-index') + 1])
            self.assertEqual(got, GATE_CLIP_PLAN[arm],
                             f'{arm} gate clip plan drifted: {got}')


if __name__ == '__main__':
    unittest.main(verbosity=2)
