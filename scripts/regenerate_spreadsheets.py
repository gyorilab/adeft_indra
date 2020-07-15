import numpy as np
import pandas as pd

group1 = pd.read_csv('../results/light_kinase_group1.csv')
group2 = pd.read_csv('../results/light_kinase_group2.csv')
group3 = pd.read_csv('../results/light_kinase_group3_v3.csv', sep=',')
group4 = pd.read_excel('../results/light_kinase_group4.xls')

df1 = group1[group1.CI_lower > 0]
df2 = group2[group2.CI_lower > 0]
df3 = group3[group3.CI_lower > 0]
df4 = group4[group4.CI_lower > 0]


df1.columns = ['gene_name', 'agent_text', 'num_stmts', 'num_entrez',
               'CI_lower', 'CI_upper', 'adeftable', 'correct',
               'what to do', 'comment']
df2['correct'] = np.nan
df2['what to do'] = np.nan
df2['comment'] = np.nan
df4.columns = ['gene_name', 'agent_text', 'num_stmts', 'num_entrez',
               'CI_lower', 'CI_upper', 'adeftable', 'correct',
               'what to do', 'comment']

df1['group'] = 1
df2['group'] = 2
df3['group'] = 3
df4['group'] = 4


set1 = pd.concat([df1, df2, df3, df4], sort=False)


df = group4.copy()
counts = pd.DataFrame(df.groupby('gene_name')['num_stmts'].sum())
counts.reset_index(inplace=True)

a = df[df.CI_lower > 0]
counts_a = pd.DataFrame(a.groupby('gene_name')['num_stmts'].sum())
counts_a.reset_index(inplace=True)

b = df[(df.CI_upper < 1) & (df.CI_lower == 0)]
counts_b = pd.DataFrame(b.groupby('gene_name')['num_stmts'].sum())
counts_b.reset_index(inplace=True)

genes = pd.DataFrame(df['gene_name']).groupby('gene_name').first().\
    reset_index()

genes = genes.merge(counts_a, how='outer', on='gene_name')
genes = genes.merge(counts_b, how='outer', on='gene_name')
genes = genes.merge(counts, how='outer', on='gene_name')
genes = genes.fillna(0)
genes.columns = ['gene_name', 'low_CI', 'high_CI', 'total']
genes['coverage'] = (genes['low_CI'] + genes['high_CI'])/genes['total']

covered_genes = genes[genes.coverage >= 0.8]
covered_genes['total_target_count'] = covered_genes\
    .apply(lambda row:
           int(np.ceil(0.8 * row.total)),
           axis=1)
covered_genes['target_count'] = covered_genes['total_target_count'] - \
    covered_genes.low_CI
covered_genes['target_count'] = covered_genes['target_count'].\
    apply(lambda x:
          0 if x < 0 else x)

c = b[~b.adeftable]
c = c[~c.gene_name.isin(covered_genes.gene_name)]
c.to_csv('light_kinase_curation_set3_g4.csv', sep=',', index=False)

groups = c.groupby('gene_name')
rows = []
for gene, group in groups:
    target_count = covered_genes[covered_genes.gene_name == gene].\
        target_count.values[0]
    if target_count == 0:
        continue
    stmt_count = 0
    for index, row in group.sort_values('num_stmts', ascending=False).iterrows():
        if stmt_count > target_count:
            break
        rows.append(row.tolist())
        stmt_count += row.num_stmts
curation_set = pd.DataFrame(rows, columns=c.columns)
# curation_set.to_csv('../results/light_kinase_group4_set2.csv')


group1_2 = pd.read_csv('../results/light_kinase_group1_set2.csv', sep=',')
group2_2 = pd.read_csv('../results/light_kinase_group2_set2.csv', sep=',')
group3_2 = pd.read_csv('../results/light_kinase_group3_set2.csv', sep=',')
group4_2 = pd.read_csv('../results/light_kinase_group4_set2.csv', sep=',')
group1_2.drop(group1_2.columns[0], axis=1, inplace=True)
group2_2.drop(group2_2.columns[0], axis=1, inplace=True)
group3_2.drop(group3_2.columns[0], axis=1, inplace=True)
group4_2.drop(group4_2.columns[0], axis=1, inplace=True)


group4_2.columns = group3_2.columns

group1_2['group'] = 1
group2_2['group'] = 2
group3_2['group'] = 3
group4_2['group'] = 4

group1_3 = pd.read_csv('../results/light_kinase_curation_set3_g1.csv', sep=',')
group2_3 = pd.read_csv('../results/light_kinase_curation_set3_g2.csv', sep=',')
group3_3 = pd.read_csv('../results/light_kinase_curation_set3_g3.csv', sep=',')
group4_3 = pd.read_csv('../results/light_kinase_curation_set3_g4.csv', sep=',')

group1_3['group'] = 1
group2_3['group'] = 2
group3_3['group'] = 3
group4_3['group'] = 4




curations = pd.concat([group1_3, group2_3, group3_3, group4_3], sort=False)
curations.to_csv('../results/light_kinase_curation_set2.csv', sep=',', index=False)

kinases = curations.gene_name.unique()
counts = curations.groupby('gene_name')['agent_text'].count()
counts = counts.sort_values(ascending=False)

curations = pd.read_csv('~/adeft_indra/results/light_kinase_curation_set2_v2.csv', sep=',')

curations['grounding_mapped'] = curations.agent_text.apply(lambda x: x in gm)
curations['mapped_to'] = curations.agent_text.apply(lambda x: gm[x] if x in gm else np.nan)
curations.to_csv('../results/light_kinase_curation_set3.csv', sep=',', index=False)
