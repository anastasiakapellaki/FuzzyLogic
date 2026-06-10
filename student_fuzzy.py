import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd
import os

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

rule10 = ctrl.Rule(
    participation['low'] &
    assignments['high'] &
    exams['high'] &
    absences['few'],
    performance['high']
)

rule11 = ctrl.Rule(
    exams['high'] &
    absences['many'],
    performance['medium']
)

rule12 = ctrl.Rule(
    participation['high'] &
    absences['many'],
    performance['medium']
)

rule13 = ctrl.Rule(
    assignments['low'] &
    exams['high'] &
    participation['high'],
    performance['medium']
)

rule14 = ctrl.Rule(
    exams['low'] &
    absences['many'],
    performance['low']
)

rule15 = ctrl.Rule(
    participation['high'] &
    assignments['high'] &
    exams['low'] &
    absences['few'],
    performance['medium']
)

rule16 = ctrl.Rule(
    participation['medium'] &
    exams['low'] &
    assignments['high'] &
    absences['few'],
    performance['medium']
)

rule17 = ctrl.Rule(
    participation['low'] &
    absences['many'],
    performance['low']
)   #decision making choice, if a student doesn't show up and he/she is not participating, performance is low regardless exams/assignments

rule18 = ctrl.Rule(
    participation['medium'] &
    assignments['high'] &
    exams['high'] &
    absences['medium'],
    performance['high']
)

rule19 = ctrl.Rule(
    participation['low'] &
    assignments['low'] &
    exams['low'] &
    absences['few'],
    performance['low']
)

rule20 = ctrl.Rule(
    participation['medium'] &
    assignments['medium'] &
    exams['medium'] &
    absences['many'],
    performance['low']
)

rule21 = ctrl.Rule(
    assignments['low'] &
    exams['low'],
    performance['low']
)



system = ctrl.ControlSystem([rule1,rule2,rule3,rule4,rule5,rule6,rule7,rule8,rule9,
                             rule10,rule11,rule12,rule13,rule14,rule15,
                             rule16,rule17,rule18,rule19,rule20, rule21])

sim = ctrl.ControlSystemSimulation(system) #initialize instance

sim.input['participation'] = 75
sim.input['assignments'] = 8
sim.input['exams'] = 9
sim.input['absences'] = 2
sim.compute()
print(sim.output['performance'])

# Run for student B
sim.input['participation'] = 30
sim.input['assignments'] = 4
sim.input['exams'] = 3
sim.input['absences'] = 15
sim.compute()
print(sim.output['performance'])

# The average student
sim.input['participation'] = 50
sim.input['assignments'] = 5
sim.input['exams'] = 5
sim.input['absences'] = 10
sim.compute()
print(f"Average student: {sim.output['performance']:.1f}")

# Silent clever student - never participates but excels in everything
sim.input['participation'] = 10
sim.input['assignments'] = 9
sim.input['exams'] = 10
sim.input['absences'] = 1
sim.compute()
print(f"Silent genius: {sim.output['performance']:.1f}")

# Participates a lot but fails 
sim.input['participation'] = 90
sim.input['assignments'] = 2
sim.input['exams'] = 2
sim.input['absences'] = 3
sim.compute()
print(f"Enthusiastic but struggling: {sim.output['performance']:.1f}")

# Good grades but never shows up
sim.input['participation'] = 20
sim.input['assignments'] = 8
sim.input['exams'] = 8
sim.input['absences'] = 18
sim.compute()
print(f"Good grades but absent: {sim.output['performance']:.1f}")

if os.path.exists('student_data.csv'):
    df = pd.read_csv('student_data.csv')
else:
    dataset = []
    for _ in range(500):
        p = np.random.randint(0, 101)
        a = np.random.randint(0, 11)
        e = np.random.randint(0, 11)
        ab = np.random.randint(0, 21)
        try:
            sim.input['participation'] = p
            sim.input['assignments'] = a
            sim.input['exams'] = e
            sim.input['absences'] = ab
            sim.compute()
            noise = np.random.normal(0, 3)
            perf = np.clip(sim.output['performance'] + noise, 0, 100)
            dataset.append([p, a, e, ab, perf])
        except:
            pass

    df = pd.DataFrame(dataset, columns=['participation', 'assignments', 'exams', 'absences', 'performance'])
    df.to_csv('student_data.csv', index=False)