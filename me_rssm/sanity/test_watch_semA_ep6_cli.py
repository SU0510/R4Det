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


def _read(rel):
    with open(os.path.join(_REPO, rel), encoding='utf-8') as handle:
        return handle.read()


def watchdog_commands():
    """Every probe invocation in the watchdog, as argv lists."""
    text = _read(os.path.join('me_rssm', 'sanity', 'watch_semA_ep6.sh'))
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


if __name__ == '__main__':
    unittest.main(verbosity=2)
