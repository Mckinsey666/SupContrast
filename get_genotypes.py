
genotypes = []
with open('genotype.txt', 'r') as file:
    for line in file:
        if 'genotype_25' in line:
            _, g = line.strip().split(': ')
            g = eval(g)
            genotypes.append(g)

print(len(genotypes))