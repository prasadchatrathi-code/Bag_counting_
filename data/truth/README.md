# Ground truth

Hand-counted events for evaluation (`mise run eval -- out/<run>/events.csv data/truth/<file>.csv`).
`time_s` is seconds from the start of the named clip; the moment the bag's centre crosses the tailgate edge.

| file | clip | window | loaded | unloaded | notes |
|---|---|---|---|---|---|
| mid_1300_first180s.csv | clips/mid_1300.mp4 (source t=5100 s, camera 12:59:59) | 0–180 s | 19 | 0 | counted from 2 fps contact sheets; one doubtful extra bag near 35.5 s was excluded. A fresh pile of static bags arrives on the dock at ~175 s. |
| late_1520_first120s.csv | clips/late_1520.mp4 (source t=12000 s, camera 14:54:59) | 0–120 s | 7 | 0 | hand-passing phase, stack above side walls. Blue drums carried past the truck at 47–52 s are not bags. Loading pauses after ~37 s. |
| early_1140_first120s.csv | clips/early_1140.mp4 (source t=300 s, camera 11:39:59) | 0–120 s | 9 | 0 | empty truck, bags handed up over the tailgate to a carrier on the bed. 81.5 s and 85.5 s verified on a 4 fps zoom. Loading pauses after ~98 s. |
