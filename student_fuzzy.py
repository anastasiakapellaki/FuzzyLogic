import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

participation = ctrl.Antecedent(np.arange(0, 101, 1), 'participation')
assignments = ctrl.Antecedent(np.arange(0, 11, 1), 'assignments')
exams = ctrl.Antecedent(np.arange(0, 11, 1), 'exams')
absences = ctrl.Antecedent(np.arange(0, 21, 1), 'absences')
participation['low'] = fuzz.trimf(participation.universe, [0, 0, 50])
participation['medium'] = fuzz.trimf(participation.universe, [25, 50, 75])
participation['high'] = fuzz.trimf(participation.universe, [50, 100, 100])
assignments['low'] = fuzz.trimf(assignments.universe, [0, 0, 5])
assignments['medium'] = fuzz.trimf(assignments.universe, [2.5, 5, 7.5])
assignments['high'] = fuzz.trimf(assignments.universe, [5, 10, 10])
exams['low'] = fuzz.trimf(exams.universe, [0, 0, 5])
exams['medium'] = fuzz.trimf(exams.universe, [2.5, 5, 7.5])
exams['high'] = fuzz.trimf(exams.universe, [5, 10, 10])
absences['few'] = fuzz.trimf(absences.universe, [0, 0, 8])
absences['medium'] = fuzz.trimf(absences.universe, [4, 10, 16])
absences['many'] = fuzz.trimf(absences.universe, [12, 20, 20])
performance = ctrl.Consequent(
    np.arange(0, 101, 1),
    'performance'
)
performance['low'] = fuzz.trimf(performance.universe, [0, 0, 50])
performance['medium'] = fuzz.trimf(performance.universe, [25, 50, 75])
performance['high'] = fuzz.trimf(performance.universe, [50, 100, 100])



rule1 = ctrl.Rule(
    participation['high'] &
    assignments['high'] &
    exams['high'] &
    absences['few'],
    performance['high']
)

rule2 = ctrl.Rule(
    participation['high'] &
    assignments['high'] &
    exams['medium'] &
    absences['few'],
    performance['high']
)

rule3 = ctrl.Rule(
    participation['medium'] &
    assignments['high'] &
    exams['high'] &
    absences['few'],
    performance['high']
)

rule4 = ctrl.Rule(
    participation['high'] &
    assignments['medium'] &
    exams['high'] &
    absences['medium'],
    performance['high']
)

rule5 = ctrl.Rule(
    participation['medium'] &
    assignments['medium'] &
    exams['medium'] &
    absences['medium'],
    performance['medium']
)

rule6 = ctrl.Rule(
    participation['low'] &
    assignments['medium'] &
    exams['medium'] &
    absences['many'],
    performance['low']
)

rule7 = ctrl.Rule(
    participation['low'] &
    assignments['low'] &
    exams['low'] &
    absences['many'],
    performance['low']
)

rule8 = ctrl.Rule(
    participation['high'] &
    assignments['low'] &
    exams['low'] &
    absences['few'],
    performance['medium']
)


rule9 = ctrl.Rule(
    participation['medium'] &
    assignments['high'] &
    exams['medium'] &
    absences['medium'],
    performance['medium']
)

