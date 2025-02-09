import matplotlib.pyplot as plt
import pandas as pd

# Data for I-BERT with and without bias shift
data = [
    {'Dataset': 'sst2', 'Mode': 'Softermax', 'Bits': 2.0, 'Accuracy': 0.4908256880733945},
    {'Dataset': 'sst2', 'Mode': 'Softermax', 'Bits': 3.0, 'Accuracy': 0.4908256880733945},
    {'Dataset': 'sst2', 'Mode': 'Softermax', 'Bits': 4.0, 'Accuracy': 0.4908256880733945},
    {'Dataset': 'sst2', 'Mode': 'Softermax', 'Bits': 5.0, 'Accuracy': 0.768348623853211},
    {'Dataset': 'sst2', 'Mode': 'Softermax', 'Bits': 6.0, 'Accuracy': 0.9059633027522935},
    {'Dataset': 'sst2', 'Mode': 'Softermax', 'Bits': 7.0, 'Accuracy': 0.9013761467889908},
    {'Dataset': 'sst2', 'Mode': 'Softermax', 'Bits': 8.0, 'Accuracy': 0.9048165137614679},
    {'Dataset': 'cola', 'Mode': 'Softermax', 'Bits': 4.0, 'Mcc': -0.028044982189654497},
    {'Dataset': 'cola', 'Mode': 'Softermax', 'Bits': 5.0, 'Mcc': 0.4491213009578182},
    {'Dataset': 'cola', 'Mode': 'Softermax', 'Bits': 6.0, 'Mcc': 0.5244371898114317},
    {'Dataset': 'cola', 'Mode': 'Softermax', 'Bits': 7.0, 'Mcc': 0.5330482808147131},
    {'Dataset': 'cola', 'Mode': 'Softermax', 'Bits': 8.0, 'Mcc': 0.5462521859735874},
    {'Dataset': 'rte', 'Mode': 'Softermax', 'Bits': 4.0, 'Accuracy': 0.5270758122743683},
    {'Dataset': 'rte', 'Mode': 'Softermax', 'Bits': 5.0, 'Accuracy': 0.5270758122743683},
    {'Dataset': 'rte', 'Mode': 'Softermax', 'Bits': 6.0, 'Accuracy': 0.5270758122743683},
    {'Dataset': 'rte', 'Mode': 'Softermax', 'Bits': 7.0, 'Accuracy': 0.5270758122743683},
    {'Dataset': 'mrpc', 'Mode': 'Softermax', 'Bits': 4.0, 'F1': 0.0},
    {'Dataset': 'mrpc', 'Mode': 'Softermax', 'Bits': 5.0, 'F1': 0.014084507042253521},
    {'Dataset': 'mrpc', 'Mode': 'Softermax', 'Bits': 6.0, 'F1': 0.6414414414414414},
    {'Dataset': 'mrpc', 'Mode': 'Softermax', 'Bits': 7.0, 'F1': 0.7958271236959762},
    {'Dataset': 'rte', 'Mode': 'Softermax', 'Bits': 8.0, 'Accuracy': 0.5451263537906137},
    {'Dataset': 'mrpc', 'Mode': 'Softermax', 'Bits': 8.0, 'F1': 0.7575221238938054},
    {'Dataset': 'sst2', 'Mode': 'I-BERT', 'Bits': 2.0, 'Accuracy': 0.4908256880733945},
    {'Dataset': 'sst2', 'Mode': 'I-BERT', 'Bits': 3.0, 'Accuracy': 0.4908256880733945},
    {'Dataset': 'sst2', 'Mode': 'I-BERT', 'Bits': 4.0, 'Accuracy': 0.4908256880733945},
    {'Dataset': 'sst2', 'Mode': 'I-BERT', 'Bits': 5.0, 'Accuracy': 0.7660550458715596},
    {'Dataset': 'sst2', 'Mode': 'I-BERT', 'Bits': 6.0, 'Accuracy': 0.9036697247706422},
    {'Dataset': 'sst2', 'Mode': 'I-BERT', 'Bits': 7.0, 'Accuracy': 0.9048165137614679},
    {'Dataset': 'sst2', 'Mode': 'I-BERT', 'Bits': 8.0, 'Accuracy': 0.9036697247706422},
    {'Dataset': 'cola', 'Mode': 'I-BERT', 'Bits': 2.0, 'Mcc': 0.0},
    {'Dataset': 'cola', 'Mode': 'I-BERT', 'Bits': 3.0, 'Mcc': 0.0},
    {'Dataset': 'cola', 'Mode': 'I-BERT', 'Bits': 4.0, 'Mcc': 0.0},
    {'Dataset': 'cola', 'Mode': 'I-BERT', 'Bits': 5.0, 'Mcc': 0.35388748299431894},
    {'Dataset': 'cola', 'Mode': 'I-BERT', 'Bits': 6.0, 'Mcc': 0.51728018358102},
    {'Dataset': 'cola', 'Mode': 'I-BERT', 'Bits': 7.0, 'Mcc': 0.5286324175580216},
    {'Dataset': 'cola', 'Mode': 'I-BERT', 'Bits': 8.0, 'Mcc': 0.5110339437874821},
    {'Dataset': 'rte', 'Mode': 'I-BERT', 'Bits': 2.0, 'Accuracy': 0.5270758122743683},
    {'Dataset': 'rte', 'Mode': 'I-BERT', 'Bits': 3.0, 'Accuracy': 0.5270758122743683},
    {'Dataset': 'rte', 'Mode': 'I-BERT', 'Bits': 4.0, 'Accuracy': 0.5270758122743683},
    {'Dataset': 'rte', 'Mode': 'I-BERT', 'Bits': 5.0, 'Accuracy': 0.5270758122743683},
    {'Dataset': 'rte', 'Mode': 'I-BERT', 'Bits': 6.0, 'Accuracy': 0.5703971119133574},
    {'Dataset': 'rte', 'Mode': 'I-BERT', 'Bits': 7.0, 'Accuracy': 0.6389891696750902},
    {'Dataset': 'rte', 'Mode': 'I-BERT', 'Bits': 8.0, 'Accuracy': 0.631768953068592},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT', 'Bits': 2.0, 'F1': 0.0},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT', 'Bits': 3.0, 'F1': 0.0},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT', 'Bits': 4.0, 'F1': 0.0},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT', 'Bits': 5.0, 'F1': 0.0},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT', 'Bits': 6.0, 'F1': 0.7412008281573499},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT', 'Bits': 7.0, 'F1': 0.8850174216027874},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT', 'Bits': 8.0, 'F1': 0.8752079866888519},
]
shift_data = [
    {'Dataset': 'sst2', 'Mode': 'I-BERT with Shift', 'Bits': 2.0, 'Accuracy': 0.4908256880733945},
    {'Dataset': 'cola', 'Mode': 'I-BERT with Shift', 'Bits': 2.0, 'Mcc': 0.0},
    {'Dataset': 'rte', 'Mode': 'I-BERT with Shift', 'Bits': 2.0, 'Accuracy': 0.5270758122743683},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT with Shift', 'Bits': 2.0, 'F1': 0.0},
    {'Dataset': 'sst2', 'Mode': 'I-BERT with Shift', 'Bits': 3.0, 'Accuracy': 0.4908256880733945},
    {'Dataset': 'cola', 'Mode': 'I-BERT with Shift', 'Bits': 3.0, 'Mcc': 0.0},
    {'Dataset': 'rte', 'Mode': 'I-BERT with Shift', 'Bits': 3.0, 'Accuracy': 0.5270758122743683},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT with Shift', 'Bits': 3.0, 'F1': 0.0},
    {'Dataset': 'sst2', 'Mode': 'I-BERT with Shift', 'Bits': 4.0, 'Accuracy': 0.5653669724770642},
    {'Dataset': 'cola', 'Mode': 'I-BERT with Shift', 'Bits': 4.0, 'Mcc': 0.37739928325901934},
    {'Dataset': 'rte', 'Mode': 'I-BERT with Shift', 'Bits': 4.0, 'Accuracy': 0.5270758122743683},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT with Shift', 'Bits': 4.0, 'F1': 0.0},
    {'Dataset': 'sst2', 'Mode': 'I-BERT with Shift', 'Bits': 5.0, 'Accuracy': 0.7993119266055045},
    {'Dataset': 'cola', 'Mode': 'I-BERT with Shift', 'Bits': 5.0, 'Mcc': 0.47429960695134216},
    {'Dataset': 'rte', 'Mode': 'I-BERT with Shift', 'Bits': 5.0, 'Accuracy': 0.5487364620938628},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT with Shift', 'Bits': 5.0, 'F1': 0.05536332179930796},
    {'Dataset': 'sst2', 'Mode': 'I-BERT with Shift', 'Bits': 6.0, 'Accuracy': 0.9048165137614679},
    {'Dataset': 'cola', 'Mode': 'I-BERT with Shift', 'Bits': 6.0, 'Mcc': 0.49660257394438423},
    {'Dataset': 'rte', 'Mode': 'I-BERT with Shift', 'Bits': 6.0, 'Accuracy': 0.5848375451263538},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT with Shift', 'Bits': 6.0, 'F1': 0.7519685039370079},
    {'Dataset': 'sst2', 'Mode': 'I-BERT with Shift', 'Bits': 7.0, 'Accuracy': 0.9002293577981652},
    {'Dataset': 'cola', 'Mode': 'I-BERT with Shift', 'Bits': 7.0, 'Mcc': 0.5201268334027794},
    {'Dataset': 'rte', 'Mode': 'I-BERT with Shift', 'Bits': 7.0, 'Accuracy': 0.6353790613718412},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT with Shift', 'Bits': 7.0, 'F1': 0.8811881188118812},
    {'Dataset': 'sst2', 'Mode': 'I-BERT with Shift', 'Bits': 8.0, 'Accuracy': 0.9025229357798165},
    {'Dataset': 'cola', 'Mode': 'I-BERT with Shift', 'Bits': 8.0, 'Mcc': 0.5023519246991239},
    {'Dataset': 'rte', 'Mode': 'I-BERT with Shift', 'Bits': 8.0, 'Accuracy': 0.6425992779783394},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT with Shift', 'Bits': 8.0, 'F1': 0.8667736757624398},

    {'Dataset': 'sst2', 'Mode': 'I-BERTV2', 'Bits': 2.0, 'Accuracy': 0.49311926605504586},
    {'Dataset': 'cola', 'Mode': 'I-BERTV2', 'Bits': 2.0, 'Mcc': 0.0},
    {'Dataset': 'rte', 'Mode': 'I-BERTV2', 'Bits': 2.0, 'Accuracy': 0.5270758122743683},
    {'Dataset': 'mrpc', 'Mode': 'I-BERTV2', 'Bits': 2.0, 'F1': 0.0},
    {'Dataset': 'sst2', 'Mode': 'I-BERTV2', 'Bits': 3.0, 'Accuracy': 0.49311926605504586},
    {'Dataset': 'cola', 'Mode': 'I-BERTV2', 'Bits': 3.0, 'Mcc': 0.018957852091478794},
    {'Dataset': 'rte', 'Mode': 'I-BERTV2', 'Bits': 3.0, 'Accuracy': 0.5270758122743683},
    {'Dataset': 'mrpc', 'Mode': 'I-BERTV2', 'Bits': 3.0, 'F1': 0.0},
    {'Dataset': 'sst2', 'Mode': 'I-BERTV2', 'Bits': 4.0, 'Accuracy': 0.8188073394495413},
    {'Dataset': 'cola', 'Mode': 'I-BERTV2', 'Bits': 4.0, 'Mcc': 0.40739928325901934},
    {'Dataset': 'rte', 'Mode': 'I-BERTV2', 'Bits': 4.0, 'Accuracy': 0.5270758122743683},
    {'Dataset': 'mrpc', 'Mode': 'I-BERTV2', 'Bits': 4.0, 'F1': 0.6299810246679317},
    {'Dataset': 'sst2', 'Mode': 'I-BERTV2', 'Bits': 5.0, 'Accuracy': 0.9013761467889908},
    {'Dataset': 'cola', 'Mode': 'I-BERTV2', 'Bits': 5.0, 'Mcc': 0.47826643579049904},
    {'Dataset': 'rte', 'Mode': 'I-BERTV2', 'Bits': 5.0, 'Accuracy': 0.555956678700361},
    {'Dataset': 'mrpc', 'Mode': 'I-BERTV2', 'Bits': 5.0, 'F1': 0.7454909819639278},
    {'Dataset': 'sst2', 'Mode': 'I-BERTV2', 'Bits': 6.0, 'Accuracy': 0.9002293577981652},
    {'Dataset': 'cola', 'Mode': 'I-BERTV2', 'Bits': 6.0, 'Mcc': 0.512703445942988},
    {'Dataset': 'rte', 'Mode': 'I-BERTV2', 'Bits': 6.0, 'Accuracy': 0.6462093862815884},
    {'Dataset': 'mrpc', 'Mode': 'I-BERTV2', 'Bits': 6.0, 'F1': 0.8301886792452831},
    {'Dataset': 'sst2', 'Mode': 'I-BERTV2', 'Bits': 7.0, 'Accuracy': 0.9094036697247706},
    {'Dataset': 'cola', 'Mode': 'I-BERTV2', 'Bits': 7.0, 'Mcc': 0.5207572018426233},
    {'Dataset': 'rte', 'Mode': 'I-BERTV2', 'Bits': 7.0, 'Accuracy': 0.6570397111913358},
    {'Dataset': 'mrpc', 'Mode': 'I-BERTV2', 'Bits': 7.0, 'F1': 0.875886524822695},
    {'Dataset': 'sst2', 'Mode': 'I-BERTV2', 'Bits': 8.0, 'Accuracy': 0.9059633027522935},
    {'Dataset': 'cola', 'Mode': 'I-BERTV2', 'Bits': 8.0, 'Mcc': 0.5046718184221465},
    {'Dataset': 'rte', 'Mode': 'I-BERTV2', 'Bits': 8.0, 'Accuracy': 0.6498194945848376},
    {'Dataset': 'mrpc', 'Mode': 'I-BERTV2', 'Bits': 8.0, 'F1': 0.8784722222222222},

    {'Dataset': 'sst2', 'Mode': 'I-BERTV2 with Shift', 'Bits': 2.0, 'Accuracy': 0.4908256880733945},
    {'Dataset': 'cola', 'Mode': 'I-BERTV2 with Shift', 'Bits': 2.0, 'Mcc': 0.0},
    {'Dataset': 'rte', 'Mode': 'I-BERTV2 with Shift', 'Bits': 2.0, 'Accuracy': 0.4657039711191336},
    {'Dataset': 'mrpc', 'Mode': 'I-BERTV2 with Shift', 'Bits': 2.0, 'F1': 0.0},
    {'Dataset': 'sst2', 'Mode': 'I-BERTV2 with Shift', 'Bits': 3.0, 'Accuracy': 0.6112385321100917},
    {'Dataset': 'cola', 'Mode': 'I-BERTV2 with Shift', 'Bits': 3.0, 'Mcc': 0.17077651041202851},
    {'Dataset': 'rte', 'Mode': 'I-BERTV2 with Shift', 'Bits': 3.0, 'Accuracy': 0.5018050541516246},
    {'Dataset': 'mrpc', 'Mode': 'I-BERTV2 with Shift', 'Bits': 3.0, 'F1': 0.007067137809187279},
    {'Dataset': 'sst2', 'Mode': 'I-BERTV2 with Shift', 'Bits': 4.0, 'Accuracy': 0.8623853211009175},
    {'Dataset': 'cola', 'Mode': 'I-BERTV2 with Shift', 'Bits': 4.0, 'Mcc': 0.4312263576038266},
    {'Dataset': 'rte', 'Mode': 'I-BERTV2 with Shift', 'Bits': 4.0, 'Accuracy': 0.516245487364621},
    {'Dataset': 'mrpc', 'Mode': 'I-BERTV2 with Shift', 'Bits': 4.0, 'F1': 0.6857039711191336},
    {'Dataset': 'sst2', 'Mode': 'I-BERTV2 with Shift', 'Bits': 5.0, 'Accuracy': 0.9013761467889908},
    {'Dataset': 'cola', 'Mode': 'I-BERTV2 with Shift', 'Bits': 5.0, 'Mcc': 0.47826643579049904},
    {'Dataset': 'rte', 'Mode': 'I-BERTV2 with Shift', 'Bits': 5.0, 'Accuracy': 0.555956678700361},
    {'Dataset': 'mrpc', 'Mode': 'I-BERTV2 with Shift', 'Bits': 5.0, 'F1': 0.7454909819639278},
    {'Dataset': 'sst2', 'Mode': 'I-BERTV2 with Shift', 'Bits': 6.0, 'Accuracy': 0.9002293577981652},
    {'Dataset': 'cola', 'Mode': 'I-BERTV2 with Shift', 'Bits': 6.0, 'Mcc': 0.512703445942988},
    {'Dataset': 'rte', 'Mode': 'I-BERTV2 with Shift', 'Bits': 6.0, 'Accuracy': 0.6462093862815884},
    {'Dataset': 'mrpc', 'Mode': 'I-BERTV2 with Shift', 'Bits': 6.0, 'F1': 0.8301886792452831},
    {'Dataset': 'sst2', 'Mode': 'I-BERTV2 with Shift', 'Bits': 7.0, 'Accuracy': 0.9094036697247706},
    {'Dataset': 'cola', 'Mode': 'I-BERTV2 with Shift', 'Bits': 7.0, 'Mcc': 0.5207572018426233},
    {'Dataset': 'rte', 'Mode': 'I-BERTV2 with Shift', 'Bits': 7.0, 'Accuracy': 0.6570397111913358},
    {'Dataset': 'mrpc', 'Mode': 'I-BERTV2 with Shift', 'Bits': 7.0, 'F1': 0.875886524822695},
    {'Dataset': 'sst2', 'Mode': 'I-BERTV2 with Shift', 'Bits': 8.0, 'Accuracy': 0.9059633027522935},
    {'Dataset': 'cola', 'Mode': 'I-BERTV2 with Shift', 'Bits': 8.0, 'Mcc': 0.5046718184221465},
    {'Dataset': 'rte', 'Mode': 'I-BERTV2 with Shift', 'Bits': 8.0, 'Accuracy': 0.6498194945848376},
    {'Dataset': 'mrpc', 'Mode': 'I-BERTV2 with Shift', 'Bits': 8.0, 'F1': 0.8784722222222222},
]
# Colorblind-friendly color palette
color_palette = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#984ea3', '#a65628', '#999999', '#e41a1c', '#dede00']

# Create a DataFrame
df = pd.DataFrame(data)

# Baseline values
baseline_values = {
    'sst2': 0.9036697247706422,
    'cola': 0.5277813760438573,
    'rte': 0.6678700361010831,
    'mrpc': 0.8972602739726028
}

# Random values
random_values = {
    'sst2': 0.4908256880733945,
    'cola': 0.0,
    'rte': 0.5270758122743683,
    'mrpc': 0.0
}

import matplotlib.pyplot as plt

# Function to ensure all bits are present with additional data included
def ensure_all_bits(df, dataset, metric):
    bits = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    modes = df['Mode'].unique()
    complete_data = []
    for mode in modes:
        df_mode = df[(df['Dataset'] == dataset) & (df['Mode'] == mode)][['Bits', metric]]
        df_mode.set_index('Bits', inplace=True)
        df_mode = df_mode.reindex(bits)
        df_mode['Mode'] = mode
        df_mode['Dataset'] = dataset
        complete_data.append(df_mode.reset_index())
    return pd.concat(complete_data)

# Data for I-BERT with and without shift
shift_data = [
    {'Dataset': 'sst2', 'Mode': 'I-BERT with Shift', 'Bits': 2.0, 'Accuracy': 0.4908256880733945},
    {'Dataset': 'cola', 'Mode': 'I-BERT with Shift', 'Bits': 2.0, 'Mcc': 0.0},
    {'Dataset': 'rte', 'Mode': 'I-BERT with Shift', 'Bits': 2.0, 'Accuracy': 0.5270758122743683},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT with Shift', 'Bits': 2.0, 'F1': 0.0},
    {'Dataset': 'sst2', 'Mode': 'I-BERT with Shift', 'Bits': 3.0, 'Accuracy': 0.4908256880733945},
    {'Dataset': 'cola', 'Mode': 'I-BERT with Shift', 'Bits': 3.0, 'Mcc': 0.0},
    {'Dataset': 'rte', 'Mode': 'I-BERT with Shift', 'Bits': 3.0, 'Accuracy': 0.5270758122743683},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT with Shift', 'Bits': 3.0, 'F1': 0.0},
    {'Dataset': 'sst2', 'Mode': 'I-BERT with Shift', 'Bits': 4.0, 'Accuracy': 0.5653669724770642},
    {'Dataset': 'cola', 'Mode': 'I-BERT with Shift', 'Bits': 4.0, 'Mcc': 0.37739928325901934},
    {'Dataset': 'rte', 'Mode': 'I-BERT with Shift', 'Bits': 4.0, 'Accuracy': 0.5270758122743683},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT with Shift', 'Bits': 4.0, 'F1': 0.0},
    {'Dataset': 'sst2', 'Mode': 'I-BERT with Shift', 'Bits': 5.0, 'Accuracy': 0.7993119266055045},
    {'Dataset': 'cola', 'Mode': 'I-BERT with Shift', 'Bits': 5.0, 'Mcc': 0.47429960695134216},
    {'Dataset': 'rte', 'Mode': 'I-BERT with Shift', 'Bits': 5.0, 'Accuracy': 0.5487364620938628},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT with Shift', 'Bits': 5.0, 'F1': 0.05536332179930796},
    {'Dataset': 'sst2', 'Mode': 'I-BERT with Shift', 'Bits': 6.0, 'Accuracy': 0.9048165137614679},
    {'Dataset': 'cola', 'Mode': 'I-BERT with Shift', 'Bits': 6.0, 'Mcc': 0.49660257394438423},
    {'Dataset': 'rte', 'Mode': 'I-BERT with Shift', 'Bits': 6.0, 'Accuracy': 0.5848375451263538},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT with Shift', 'Bits': 6.0, 'F1': 0.7519685039370079},
    {'Dataset': 'sst2', 'Mode': 'I-BERT with Shift', 'Bits': 7.0, 'Accuracy': 0.9002293577981652},
    {'Dataset': 'cola', 'Mode': 'I-BERT with Shift', 'Bits': 7.0, 'Mcc': 0.5201268334027794},
    {'Dataset': 'rte', 'Mode': 'I-BERT with Shift', 'Bits': 7.0, 'Accuracy': 0.6353790613718412},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT with Shift', 'Bits': 7.0, 'F1': 0.8811881188118812},
    {'Dataset': 'sst2', 'Mode': 'I-BERT with Shift', 'Bits': 8.0, 'Accuracy': 0.9025229357798165},
    {'Dataset': 'cola', 'Mode': 'I-BERT with Shift', 'Bits': 8.0, 'Mcc': 0.5023519246991239},
    {'Dataset': 'rte', 'Mode': 'I-BERT with Shift', 'Bits': 8.0, 'Accuracy': 0.6425992779783394},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT with Shift', 'Bits': 8.0, 'F1': 0.8667736757624398},

    {'Dataset': 'sst2', 'Mode': 'I-BERTV2', 'Bits': 2.0, 'Accuracy': 0.49311926605504586},
    {'Dataset': 'cola', 'Mode': 'I-BERTV2', 'Bits': 2.0, 'Mcc': 0.0},
    {'Dataset': 'rte', 'Mode': 'I-BERTV2', 'Bits': 2.0, 'Accuracy': 0.5270758122743683},
    {'Dataset': 'mrpc', 'Mode': 'I-BERTV2', 'Bits': 2.0, 'F1': 0.0},
    {'Dataset': 'sst2', 'Mode': 'I-BERTV2', 'Bits': 3.0, 'Accuracy': 0.49311926605504586},
    {'Dataset': 'cola', 'Mode': 'I-BERTV2', 'Bits': 3.0, 'Mcc': 0.018957852091478794},
    {'Dataset': 'rte', 'Mode': 'I-BERTV2', 'Bits': 3.0, 'Accuracy': 0.5270758122743683},
    {'Dataset': 'mrpc', 'Mode': 'I-BERTV2', 'Bits': 3.0, 'F1': 0.0},
    {'Dataset': 'sst2', 'Mode': 'I-BERTV2', 'Bits': 4.0, 'Accuracy': 0.8188073394495413},
    {'Dataset': 'cola', 'Mode': 'I-BERTV2', 'Bits': 4.0, 'Mcc': 0.40739928325901934},
    {'Dataset': 'rte', 'Mode': 'I-BERTV2', 'Bits': 4.0, 'Accuracy': 0.5270758122743683},
    {'Dataset': 'mrpc', 'Mode': 'I-BERTV2', 'Bits': 4.0, 'F1': 0.6299810246679317},
    {'Dataset': 'sst2', 'Mode': 'I-BERTV2', 'Bits': 5.0, 'Accuracy': 0.9013761467889908},
    {'Dataset': 'cola', 'Mode': 'I-BERTV2', 'Bits': 5.0, 'Mcc': 0.47826643579049904},
    {'Dataset': 'rte', 'Mode': 'I-BERTV2', 'Bits': 5.0, 'Accuracy': 0.555956678700361},
    {'Dataset': 'mrpc', 'Mode': 'I-BERTV2', 'Bits': 5.0, 'F1': 0.7454909819639278},
    {'Dataset': 'sst2', 'Mode': 'I-BERTV2', 'Bits': 6.0, 'Accuracy': 0.9002293577981652},
    {'Dataset': 'cola', 'Mode': 'I-BERTV2', 'Bits': 6.0, 'Mcc': 0.512703445942988},
    {'Dataset': 'rte', 'Mode': 'I-BERTV2', 'Bits': 6.0, 'Accuracy': 0.6462093862815884},
    {'Dataset': 'mrpc', 'Mode': 'I-BERTV2', 'Bits': 6.0, 'F1': 0.8301886792452831},
    {'Dataset': 'sst2', 'Mode': 'I-BERTV2', 'Bits': 7.0, 'Accuracy': 0.9094036697247706},
    {'Dataset': 'cola', 'Mode': 'I-BERTV2', 'Bits': 7.0, 'Mcc': 0.5207572018426233},
    {'Dataset': 'rte', 'Mode': 'I-BERTV2', 'Bits': 7.0, 'Accuracy': 0.6570397111913358},
    {'Dataset': 'mrpc', 'Mode': 'I-BERTV2', 'Bits': 7.0, 'F1': 0.875886524822695},
    {'Dataset': 'sst2', 'Mode': 'I-BERTV2', 'Bits': 8.0, 'Accuracy': 0.9059633027522935},
    {'Dataset': 'cola', 'Mode': 'I-BERTV2', 'Bits': 8.0, 'Mcc': 0.5046718184221465},
    {'Dataset': 'rte', 'Mode': 'I-BERTV2', 'Bits': 8.0, 'Accuracy': 0.6498194945848376},
    {'Dataset': 'mrpc', 'Mode': 'I-BERTV2', 'Bits': 8.0, 'F1': 0.8784722222222222},

    {'Dataset': 'sst2', 'Mode': 'I-BERTV2 with Shift', 'Bits': 2.0, 'Accuracy': 0.4908256880733945},
    {'Dataset': 'cola', 'Mode': 'I-BERTV2 with Shift', 'Bits': 2.0, 'Mcc': 0.0},
    {'Dataset': 'rte', 'Mode': 'I-BERTV2 with Shift', 'Bits': 2.0, 'Accuracy': 0.4657039711191336},
    {'Dataset': 'mrpc', 'Mode': 'I-BERTV2 with Shift', 'Bits': 2.0, 'F1': 0.0},
    {'Dataset': 'sst2', 'Mode': 'I-BERTV2 with Shift', 'Bits': 3.0, 'Accuracy': 0.6112385321100917},
    {'Dataset': 'cola', 'Mode': 'I-BERTV2 with Shift', 'Bits': 3.0, 'Mcc': 0.17077651041202851},
    {'Dataset': 'rte', 'Mode': 'I-BERTV2 with Shift', 'Bits': 3.0, 'Accuracy': 0.5018050541516246},
    {'Dataset': 'mrpc', 'Mode': 'I-BERTV2 with Shift', 'Bits': 3.0, 'F1': 0.007067137809187279},
    {'Dataset': 'sst2', 'Mode': 'I-BERTV2 with Shift', 'Bits': 4.0, 'Accuracy': 0.8623853211009175},
    {'Dataset': 'cola', 'Mode': 'I-BERTV2 with Shift', 'Bits': 4.0, 'Mcc': 0.4312263576038266},
    {'Dataset': 'rte', 'Mode': 'I-BERTV2 with Shift', 'Bits': 4.0, 'Accuracy': 0.516245487364621},
    {'Dataset': 'mrpc', 'Mode': 'I-BERTV2 with Shift', 'Bits': 4.0, 'F1': 0.6857039711191336},
    {'Dataset': 'sst2', 'Mode': 'I-BERTV2 with Shift', 'Bits': 5.0, 'Accuracy': 0.9013761467889908},
    {'Dataset': 'cola', 'Mode': 'I-BERTV2 with Shift', 'Bits': 5.0, 'Mcc': 0.47826643579049904},
    {'Dataset': 'rte', 'Mode': 'I-BERTV2 with Shift', 'Bits': 5.0, 'Accuracy': 0.555956678700361},
    {'Dataset': 'mrpc', 'Mode': 'I-BERTV2 with Shift', 'Bits': 5.0, 'F1': 0.7454909819639278},
    {'Dataset': 'sst2', 'Mode': 'I-BERTV2 with Shift', 'Bits': 6.0, 'Accuracy': 0.9002293577981652},
    {'Dataset': 'cola', 'Mode': 'I-BERTV2 with Shift', 'Bits': 6.0, 'Mcc': 0.512703445942988},
    {'Dataset': 'rte', 'Mode': 'I-BERTV2 with Shift', 'Bits': 6.0, 'Accuracy': 0.6462093862815884},
    {'Dataset': 'mrpc', 'Mode': 'I-BERTV2 with Shift', 'Bits': 6.0, 'F1': 0.8301886792452831},
    {'Dataset': 'sst2', 'Mode': 'I-BERTV2 with Shift', 'Bits': 7.0, 'Accuracy': 0.9094036697247706},
    {'Dataset': 'cola', 'Mode': 'I-BERTV2 with Shift', 'Bits': 7.0, 'Mcc': 0.5207572018426233},
    {'Dataset': 'rte', 'Mode': 'I-BERTV2 with Shift', 'Bits': 7.0, 'Accuracy': 0.6570397111913358},
    {'Dataset': 'mrpc', 'Mode': 'I-BERTV2 with Shift', 'Bits': 7.0, 'F1': 0.875886524822695},
    {'Dataset': 'sst2', 'Mode': 'I-BERTV2 with Shift', 'Bits': 8.0, 'Accuracy': 0.9059633027522935},
    {'Dataset': 'cola', 'Mode': 'I-BERTV2 with Shift', 'Bits': 8.0, 'Mcc': 0.5046718184221465},
    {'Dataset': 'rte', 'Mode': 'I-BERTV2 with Shift', 'Bits': 8.0, 'Accuracy': 0.6498194945848376},
    {'Dataset': 'mrpc', 'Mode': 'I-BERTV2 with Shift', 'Bits': 8.0, 'F1': 0.8784722222222222},
]

# Extend the original I-BERT DataFrame with shift data
df_shift = pd.DataFrame(shift_data)
df_ibert = df
df_comparison = pd.concat([df_ibert, df_shift], ignore_index=True)

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(30, 20))  # Increase figure size for presentation

# Define larger font sizes and line widths
title_fontsize = 36
label_fontsize = 32
tick_fontsize = 28
legend_fontsize = 28
line_width = 8

# Plot for sst2
ax = axs[0, 0]
df_filtered = ensure_all_bits(df_comparison, 'sst2', 'Accuracy')
for i, mode in enumerate(df_filtered['Mode'].unique()):
    df_mode = df_filtered[df_filtered['Mode'] == mode]
    linestyle = "solid"
    # if "I-BERTV2 with Shift" in mode:
    #     # continue
    if "Softermax" in mode:
        linestyle = "solid"
        line_width = 16
    elif "V2" not in mode:
        linestyle = "solid"
        line_width = 8
    else:
        linestyle = "solid"
        line_width = 8
    linestyle = None
    ax.plot(df_mode['Bits'], df_mode['Accuracy'], label=mode, marker='o', color=color_palette[i % len(color_palette)], linewidth=line_width, linestyle=linestyle, markersize=16)
line_width = 8
ax.axhline(y=baseline_values['sst2'], color='r', linestyle='--', label='Baseline', linewidth=line_width, alpha=0.5)
ax.axhline(y=random_values['sst2'], color='black', linestyle='--', label='Random Guess', linewidth=line_width, alpha=0.5)
ax.set_xticks(df_filtered['Bits'].unique())
ax.set_xlabel('Bits', fontsize=label_fontsize)
ax.set_ylabel('Accuracy', fontsize=label_fontsize)
ax.set_title('sst2 - Accuracy vs Bits', fontsize=title_fontsize)
ax.legend(fontsize=legend_fontsize)
ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
ax.grid(True)

# Plot for cola
ax = axs[0, 1]
df_filtered = ensure_all_bits(df_comparison, 'cola', 'Mcc')
for i, mode in enumerate(df_filtered['Mode'].unique()):
    df_mode = df_filtered[df_filtered['Mode'] == mode]
    linestyle = "solid"
    # if "V2" not in mode:
    #     linestyle = "dotted"
    # else:
    #     linestyle = "solid"
    ax.plot(df_mode['Bits'], df_mode['Mcc'], label=mode, marker='o', color=color_palette[i % len(color_palette)], linewidth=line_width, linestyle=linestyle)
ax.axhline(y=baseline_values['cola'], color='r', linestyle='--', label='Baseline', linewidth=line_width)
ax.axhline(y=random_values['cola'], color='black', linestyle='--', label='Random Guess', linewidth=line_width)
ax.set_xticks(df_filtered['Bits'].unique())
ax.set_xlabel('Bits', fontsize=label_fontsize)
ax.set_ylabel('Mcc', fontsize=label_fontsize)
ax.set_title('cola - Mcc vs Bits', fontsize=title_fontsize)
ax.legend(fontsize=legend_fontsize)
ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
ax.grid(True)

# Plot for rte
ax = axs[1, 0]
df_filtered = ensure_all_bits(df_comparison, 'rte', 'Accuracy')
for i, mode in enumerate(df_filtered['Mode'].unique()):
    df_mode = df_filtered[df_filtered['Mode'] == mode]
    ax.plot(df_mode['Bits'], df_mode['Accuracy'], label=mode, marker='o', color=color_palette[i % len(color_palette)], linewidth=line_width)
ax.axhline(y=baseline_values['rte'], color='r', linestyle='--', label='Baseline', linewidth=line_width)
ax.axhline(y=random_values['rte'], color='black', linestyle='--', label='Random Guess', linewidth=line_width)
ax.set_xticks(df_filtered['Bits'].unique())
ax.set_xlabel('Bits', fontsize=label_fontsize)
ax.set_ylabel('Accuracy', fontsize=label_fontsize)
ax.set_title('rte - Accuracy vs Bits', fontsize=title_fontsize)
ax.legend(fontsize=legend_fontsize)
ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
ax.grid(True)

# Plot for mrpc
ax = axs[1, 1]
df_filtered = ensure_all_bits(df_comparison, 'mrpc', 'F1')
for i, mode in enumerate(df_filtered['Mode'].unique()):
    df_mode = df_filtered[df_filtered['Mode'] == mode]
    ax.plot(df_mode['Bits'], df_mode['F1'], label=mode, marker='o', color=color_palette[i % len(color_palette)], linewidth=line_width)
ax.axhline(y=baseline_values['mrpc'], color='r', linestyle='--', label='Baseline', linewidth=line_width)
ax.axhline(y=random_values['mrpc'], color='black', linestyle='--', label='Random Guess', linewidth=line_width)
ax.set_xticks(df_filtered['Bits'].unique())
ax.set_xlabel('Bits', fontsize=label_fontsize)
ax.set_ylabel('F1', fontsize=label_fontsize)
ax.set_title('mrpc - F1 vs Bits', fontsize=title_fontsize)
ax.legend(fontsize=legend_fontsize)
ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
ax.grid(True)


# Adjust layout
plt.tight_layout()
plt.savefig('plot.png')
print("Plots have been saved to bias_shift_analysis.png")
