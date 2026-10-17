# Dataset validation summary

Files: 210; demos: 10500; steps: 1230842; mean steps per demo: 117.2

| Status | Files |
| --- | --- |
| VALID_WITH_WARNINGS | 210 |

| Issue (level:code) | Occurrences | Files |
| --- | --- | --- |
| warning:alias_stored_as_copy | 24 | 6 |
| warning:exceeds_env_horizon | 12 | 12 |
| warning:exceeds_eval_horizon | 20 | 20 |
| warning:no_provenance | 210 | 210 |
| warning:replay_inconsistent | 35 | 30 |
| warning:terminal_next_obs_copy | 10500 | 210 |

Replay: 595/630 demos reach success exactly at their last step under the current code; max object deviation 0.0853 m (median 0.00002 m).

Action sequences shared across tasks: 0
