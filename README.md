# frontier-interpolation-eg
London Intervarsity Case Competition

REFER TO MAIN.PY FOR THE SIMULATION

Atlas 4.4's parameters were created using a constrained interpolation algorithm that positions the new model between the two observed system configurations in the case. our mathematical model calculates the smallest shift required from Atlas 4.3 towards Atlas FC so that all stated fairness thresholds are satisfied simultaneously. for each indicator (such as average false-positive rate, long-tenure false-positive rate, redundancy over-prediction, and the career-break odds ratio) the algorithm calculates how far the model must move toward the calibrated version for the metric to fall within the target ranges. each fairness metric may require a different level of adjustment to meet its target. the algorithm therefore identifies the metrics that requires the largest change to reach its required threshold. that calculated level of adjustment is then applied across the model, ensuring that each target is met while keeping the system performance as close as possible to Atlas 4.3.
