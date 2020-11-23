"""
This module contains information taken from Disease Ontolgy (https://disease-ontology.org/) to classify cancers based on their primary site and histology
"""

codes_mapping = {
    "hepatocellular_carcinoma": "684",
    "liver_cancer": "3571",
    "gastrointestinal_system_cancer": "3119", 
    "colon_adenocarcinoma": "234",
    "colon_carcinoma": "1520",
    "colon_cancer": "219",
    "colorectal_cancer": "9256",
    "intestinal_cancer": "10155",
    "esophageal_carcinoma": "1107",
    "stomach_cancer": "10534",
    "biliary_tract_cancer": "4607",
    
    "integumentary_system_cancer": "0060122",
    
    "breast_ductal_carcinoma": "3007",
    "breast_cancer": "1612",
    
    "ovarian_cancer": "2394",
    "endometrial_cancer": "1380",
    "cervical_cancer": "4362",
    "female_reproductive_organ_cancer": "120",
    "prostate_adenocarcinoma": "2526",
    "male_reproductive_organ_cancer": "3856",
    "reproductive_organ_cancer": "193",
    
    "clear_cell_renal_cell_carcinoma": "4467",
    "kidney_cancer": "263",
    "urinary_bladder_cancer": "11054",
    "urinary_system_cancer": "3996",
    
    "pancreatic_ductal_carcinoma": "3587",
    "pancreatic_cancer": "1793",
    "thyroid_gland_cancer": "1781",
    "adrenal_gland_cancer": "3953",
    
    "lung_adenocarcinoma": "3910",
    "lung_squamous_cell_carcinoma": "3907",
    "lung_cancer": "1324",
    "respiratory_system_cancer": "0050615",
    
    "brain_cancer": "1319",
    "central_nervous_system_cancer": "3620",
    "nervous_system_cancer": "3093",
    "brain_glioma": "0060108",
    
    "immune_system_cancer": "0060083",
    "acute_myeloid_leukaemia": "9119",
    #"myeloid_leukemia": "8692",
    "leukemia": "1240",
    "lymphoma": "0060058",
    "chronic_lymphocytic_leukemia": "1040",
    "T_cell_acute_lymphoblastic_leukemia": "5603",
    "B_cell_acute_lymphoblastic_leukemia": "0080638",
    "diffuse_large_B_cell_lymphoma": "0050745",
    "lymphoid_leukemia": "1037",
    
    "connective_tissue_cancer": "201",
    
    "carcinoma": "305",
    "squamous_cell_carcinoma": "1749",
    "adenocarcinoma": "299",
    
    "malignant_glioma": "3070",
    "astrocytoma": "3069",
    
    "disease_of_cellular_proliferation": "14566",
    "benign_neoplasm": "0060072"    
    
}

hierarchy_children = {
    "liver_cancer": ["hepatocellular_carcinoma"],
    "gastrointestinal_system_cancer": ["liver_cancer", "intestinal_cancer", "esophageal_carcinoma", "stomach_cancer", "biliary_tract_cancer"],
    "intestinal_cancer": ["colorectal_cancer"],
    "colorectal_cancer": ["colon_cancer"],
    "colon_cancer": ["colon_carcinoma"],
    "colon_carcinoma": ["colon_adenocarcinoma"],
    
    "breast_cancer": ["breast_ductal_carcinoma"],
    "thoracic_cancer": ["breast_cancer", "respiratory_system_cancer"],
    
    "female_reproductive_organ_cancer": ["ovarian_cancer", "cervical_cancer", "endometrial_cancer"],
    "male_reproductive_organ_cancer": ["prostate_adenocarcinoma"],
    "reproductive_organ_cancer": ["female_reproductive_organ_cancer", "male_reproductive_organ_cancer"],
    
    "kidney_cancer": ["clear_cell_renal_cell_carcinoma"],
    "urinary_system_cancer": ["kidney_cancer", "urinary_bladder_cancer"],
    
    "pancreatic_cancer": ["pancreatic_ductal_carcinoma"],
    "endocrine_gland_cancer": ["pancreatic_cancer", "thyroid_gland_cancer", "adrenal_gland_cancer"],
    
    "lung_cancer": ["lung_adenocarcinoma", "lung_squamous_cell_carcinoma"],
    "respiratory_system_cancer": ["lung_cancer"],
    
    "central_nervous_system_cancer": ["brain_cancer"],
    "nervous_system_cancer": ["central_nervous_system_cancer"],
    "brain_cancer": ["brain_glioma"],

    "immune_system_cancer": ["lymphoma", "leukemia"],
    "leukemia": ["acute_myeloid_leukaemia", "lymphoid_leukemia", "chronic_lymphocytic_leukemia"],
    "lymphoid_leukemia": ["T_cell_acute_lymphoblastic_leukemia", "B_cell_acute_lymphoblastic_leukemia"],
    "lymphoma": ["diffuse_large_B_cell_lymphoma"],
    
    "cancer": ["site_subtype_cancer", "histological_subtype_cancer"],
    "site_subtype_cancer": ["gastrointestinal_system_cancer", "integumentary_system_cancer", "immune_system_cancer", 
               "thoracic_cancer", "reproductive_organ_cancer", "endocrine_gland_cancer",
               "urinary_system_cancer", "nervous_system_cancer", "connective_tissue_cancer"],
    
    "histological_subtype_cancer": ["carcinoma", "malignant_glioma"],
    "carcinoma": ["squamous_cell_carcinoma", "adenocarcinoma"],
    "malignant_glioma": ["astrocytoma"],
    
    "disease_of_cellular_proliferation": ["cancer", "benign_neoplasm"],
}

hierarchy_ancestors = {'hepatocellular_carcinoma': {'cancer',
  'disease_of_cellular_proliferation',
  'gastrointestinal_system_cancer',
  'liver_cancer',
  'site_subtype_cancer'},
 'liver_cancer': {'cancer',
  'disease_of_cellular_proliferation',
  'gastrointestinal_system_cancer',
  'site_subtype_cancer'},
 'gastrointestinal_system_cancer': {'cancer',
  'disease_of_cellular_proliferation',
  'site_subtype_cancer'},
 'colon_adenocarcinoma': {'cancer',
  'colon_cancer',
  'colon_carcinoma',
  'colorectal_cancer',
  'disease_of_cellular_proliferation',
  'gastrointestinal_system_cancer',
  'intestinal_cancer',
  'site_subtype_cancer'},
 'colon_carcinoma': {'cancer',
  'colon_cancer',
  'colorectal_cancer',
  'disease_of_cellular_proliferation',
  'gastrointestinal_system_cancer',
  'intestinal_cancer',
  'site_subtype_cancer'},
 'colon_cancer': {'cancer',
  'colorectal_cancer',
  'disease_of_cellular_proliferation',
  'gastrointestinal_system_cancer',
  'intestinal_cancer',
  'site_subtype_cancer'},
 'colorectal_cancer': {'cancer',
  'disease_of_cellular_proliferation',
  'gastrointestinal_system_cancer',
  'intestinal_cancer',
  'site_subtype_cancer'},
 'intestinal_cancer': {'cancer',
  'disease_of_cellular_proliferation',
  'gastrointestinal_system_cancer',
  'site_subtype_cancer'},
 'esophageal_carcinoma': {'cancer',
  'disease_of_cellular_proliferation',
  'gastrointestinal_system_cancer',
  'site_subtype_cancer'},
 'stomach_cancer': {'cancer',
  'disease_of_cellular_proliferation',
  'gastrointestinal_system_cancer',
  'site_subtype_cancer'},
 'biliary_tract_cancer': {'cancer',
  'disease_of_cellular_proliferation',
  'gastrointestinal_system_cancer',
  'site_subtype_cancer'},
 'integumentary_system_cancer': {'cancer',
  'disease_of_cellular_proliferation',
  'site_subtype_cancer'},
 'breast_ductal_carcinoma': {'breast_cancer',
  'cancer',
  'disease_of_cellular_proliferation',
  'site_subtype_cancer',
  'thoracic_cancer'},
 'breast_cancer': {'cancer',
  'disease_of_cellular_proliferation',
  'site_subtype_cancer',
  'thoracic_cancer'},
 'ovarian_cancer': {'cancer',
  'disease_of_cellular_proliferation',
  'female_reproductive_organ_cancer',
  'reproductive_organ_cancer',
  'site_subtype_cancer'},
 'endometrial_cancer': {'cancer',
  'disease_of_cellular_proliferation',
  'female_reproductive_organ_cancer',
  'reproductive_organ_cancer',
  'site_subtype_cancer'},
 'cervical_cancer': {'cancer',
  'disease_of_cellular_proliferation',
  'female_reproductive_organ_cancer',
  'reproductive_organ_cancer',
  'site_subtype_cancer'},
 'female_reproductive_organ_cancer': {'cancer',
  'disease_of_cellular_proliferation',
  'reproductive_organ_cancer',
  'site_subtype_cancer'},
 'prostate_adenocarcinoma': {'cancer',
  'disease_of_cellular_proliferation',
  'male_reproductive_organ_cancer',
  'reproductive_organ_cancer',
  'site_subtype_cancer'},
 'male_reproductive_organ_cancer': {'cancer',
  'disease_of_cellular_proliferation',
  'reproductive_organ_cancer',
  'site_subtype_cancer'},
 'reproductive_organ_cancer': {'cancer',
  'disease_of_cellular_proliferation',
  'site_subtype_cancer'},
 'clear_cell_renal_cell_carcinoma': {'cancer',
  'disease_of_cellular_proliferation',
  'kidney_cancer',
  'site_subtype_cancer',
  'urinary_system_cancer'},
 'kidney_cancer': {'cancer',
  'disease_of_cellular_proliferation',
  'site_subtype_cancer',
  'urinary_system_cancer'},
 'urinary_bladder_cancer': {'cancer',
  'disease_of_cellular_proliferation',
  'site_subtype_cancer',
  'urinary_system_cancer'},
 'urinary_system_cancer': {'cancer',
  'disease_of_cellular_proliferation',
  'site_subtype_cancer'},
 'pancreatic_ductal_carcinoma': {'cancer',
  'disease_of_cellular_proliferation',
  'endocrine_gland_cancer',
  'pancreatic_cancer',
  'site_subtype_cancer'},
 'pancreatic_cancer': {'cancer',
  'disease_of_cellular_proliferation',
  'endocrine_gland_cancer',
  'site_subtype_cancer'},
 'thyroid_gland_cancer': {'cancer',
  'disease_of_cellular_proliferation',
  'endocrine_gland_cancer',
  'site_subtype_cancer'},
 'adrenal_gland_cancer': {'cancer',
  'disease_of_cellular_proliferation',
  'endocrine_gland_cancer',
  'site_subtype_cancer'},
 'lung_adenocarcinoma': {'cancer',
  'disease_of_cellular_proliferation',
  'lung_cancer',
  'respiratory_system_cancer',
  'site_subtype_cancer',
  'thoracic_cancer'},
 'lung_squamous_cell_carcinoma': {'cancer',
  'disease_of_cellular_proliferation',
  'lung_cancer',
  'respiratory_system_cancer',
  'site_subtype_cancer',
  'thoracic_cancer'},
 'lung_cancer': {'cancer',
  'disease_of_cellular_proliferation',
  'respiratory_system_cancer',
  'site_subtype_cancer',
  'thoracic_cancer'},
 'respiratory_system_cancer': {'cancer',
  'disease_of_cellular_proliferation',
  'site_subtype_cancer',
  'thoracic_cancer'},
 'brain_cancer': {'cancer',
  'central_nervous_system_cancer',
  'disease_of_cellular_proliferation',
  'nervous_system_cancer',
  'site_subtype_cancer'},
 'central_nervous_system_cancer': {'cancer',
  'disease_of_cellular_proliferation',
  'nervous_system_cancer',
  'site_subtype_cancer'},
 'nervous_system_cancer': {'cancer',
  'disease_of_cellular_proliferation',
  'site_subtype_cancer'},
 'brain_glioma': {'brain_cancer',
  'cancer',
  'central_nervous_system_cancer',
  'disease_of_cellular_proliferation',
  'nervous_system_cancer',
  'site_subtype_cancer'},
 'immune_system_cancer': {'cancer',
  'disease_of_cellular_proliferation',
  'site_subtype_cancer'},
 'acute_myeloid_leukaemia': {'cancer',
  'disease_of_cellular_proliferation',
  'immune_system_cancer',
  'leukemia',
  'site_subtype_cancer'},
 'leukemia': {'cancer',
  'disease_of_cellular_proliferation',
  'immune_system_cancer',
  'site_subtype_cancer'},
 'lymphoma': {'cancer',
  'disease_of_cellular_proliferation',
  'immune_system_cancer',
  'site_subtype_cancer'},
 'chronic_lymphocytic_leukemia': {'cancer',
  'disease_of_cellular_proliferation',
  'immune_system_cancer',
  'leukemia',
  'site_subtype_cancer'},
 'T_cell_acute_lymphoblastic_leukemia': {'cancer',
  'disease_of_cellular_proliferation',
  'immune_system_cancer',
  'leukemia',
  'lymphoid_leukemia',
  'site_subtype_cancer'},
 'B_cell_acute_lymphoblastic_leukemia': {'cancer',
  'disease_of_cellular_proliferation',
  'immune_system_cancer',
  'leukemia',
  'lymphoid_leukemia',
  'site_subtype_cancer'},
 'diffuse_large_B_cell_lymphoma': {'cancer',
  'disease_of_cellular_proliferation',
  'immune_system_cancer',
  'lymphoma',
  'site_subtype_cancer'},
 'lymphoid_leukemia': {'cancer',
  'disease_of_cellular_proliferation',
  'immune_system_cancer',
  'leukemia',
  'site_subtype_cancer'},
 'connective_tissue_cancer': {'cancer',
  'disease_of_cellular_proliferation',
  'site_subtype_cancer'},
 'carcinoma': {'cancer',
  'disease_of_cellular_proliferation',
  'histological_subtype_cancer'},
 'squamous_cell_carcinoma': {'cancer',
  'carcinoma',
  'disease_of_cellular_proliferation',
  'histological_subtype_cancer'},
 'adenocarcinoma': {'cancer',
  'carcinoma',
  'disease_of_cellular_proliferation',
  'histological_subtype_cancer'},
 'malignant_glioma': {'cancer',
  'disease_of_cellular_proliferation',
  'histological_subtype_cancer'},
 'astrocytoma': {'cancer',
  'disease_of_cellular_proliferation',
  'histological_subtype_cancer',
  'malignant_glioma'},
 'disease_of_cellular_proliferation': set(),
 'benign_neoplasm': {'disease_of_cellular_proliferation'},
 'site_subtype_cancer': {'cancer', 'disease_of_cellular_proliferation'},
 'histological_subtype_cancer': {'cancer',
  'disease_of_cellular_proliferation'},
 'thoracic_cancer': {'cancer',
  'disease_of_cellular_proliferation',
  'site_subtype_cancer'},
 'endocrine_gland_cancer': {'cancer',
  'disease_of_cellular_proliferation',
  'site_subtype_cancer'},
 'cancer': {'disease_of_cellular_proliferation'}}

sorted_cancer_subtypes = ['hepatocellular_carcinoma',
 'liver_cancer',
 'colon_adenocarcinoma',
 'colon_carcinoma',
 'colon_cancer',
 'colorectal_cancer',
 'intestinal_cancer',
 'esophageal_carcinoma',
 'stomach_cancer',
 'biliary_tract_cancer',
 'gastrointestinal_system_cancer',
 'integumentary_system_cancer',
 'diffuse_large_B_cell_lymphoma',
 'lymphoma',
 'acute_myeloid_leukaemia',
 'T_cell_acute_lymphoblastic_leukemia',
 'B_cell_acute_lymphoblastic_leukemia',
 'lymphoid_leukemia',
 'chronic_lymphocytic_leukemia',
 'leukemia',
 'immune_system_cancer',
 'breast_ductal_carcinoma',
 'breast_cancer',
 'lung_adenocarcinoma',
 'lung_squamous_cell_carcinoma',
 'lung_cancer',
 'respiratory_system_cancer',
 'thoracic_cancer',
 'ovarian_cancer',
 'cervical_cancer',
 'endometrial_cancer',
 'female_reproductive_organ_cancer',
 'prostate_adenocarcinoma',
 'male_reproductive_organ_cancer',
 'reproductive_organ_cancer',
 'pancreatic_ductal_carcinoma',
 'pancreatic_cancer',
 'thyroid_gland_cancer',
 'adrenal_gland_cancer',
 'endocrine_gland_cancer',
 'clear_cell_renal_cell_carcinoma',
 'kidney_cancer',
 'urinary_bladder_cancer',
 'urinary_system_cancer',
 'brain_glioma',
 'brain_cancer',
 'central_nervous_system_cancer',
 'nervous_system_cancer',
 'connective_tissue_cancer',
 'squamous_cell_carcinoma',
 'adenocarcinoma',
 'carcinoma',
 'astrocytoma',
 'malignant_glioma',
 'cancer',
 'benign_neoplasm']



def get_descendants(node):
    if node not in hierarchy_children.keys():
        return [node]
    output = []
    nodes = hierarchy_children[node]
    output.extend(nodes)
    for n in nodes:
        output.extend(get_descendants(n))
    return output

def categorical_encoding(site_subtype, hist_subtype):
    categories = set()
    if len(site_subtype)>0:
        categories.update(hierarchy_ancestors[site_subtype])
        categories.add(site_subtype)
    if len(hist_subtype)>0:
        categories.update(hierarchy_ancestors[hist_subtype])
        categories.add(hist_subtype)
    encoding = [1 if s in categories else 0 for s in sorted_cancer_subtypes]
    return encoding



def map_site_subtype(primary_site, site_subtype_1, site_subtype_2, site_subtype_3, primary_histology, histology_subtype_1, histology_subtype_2, histology_subtype_3):
    benign_subtypes_1 = ["angiomyolipoma", "neoplasm", "chondroblastoma"]
    if histology_subtype_1 in benign_subtypes_1:
        return "benign_neoplasm"
    benign_s1_approx = ["ganglioglioma", "adenoma", "benign"]
    if any(substring in histology_subtype_1 for substring in benign_s1_approx):
        return "benign_neoplasm"
    benign_s2_approx = ["giant", "benign"]
    if any(substring in histology_subtype_2 for substring in benign_s2_approx):
        return "benign_neoplasm"
    
    if primary_site == "liver":
        if histology_subtype_1 == "hepatocellular_carcinoma":
            return "hepatocellular_carcinoma"
        return "liver_cancer"
    
    if primary_site == "large_intestine":
        if site_subtype_1 == "colon":
            if primary_histology == "carcinoma":
                if histology_subtype_1 == "adenocarcinoma":
                    return "colon_adenocarcinoma"
                return "colon_carcinoma"
            return "colon_cancer"
        return "intestinal_cancer"
    if primary_site == "small_intestine":
        return "intestinal_cancer"
    if primary_site == "stomach":
        return "stomach_cancer"
    if primary_site == "oesophagus":
        return "esophageal_carcinoma"
    if primary_site == "biliary_tract":
        return "biliary_tract_cancer"
    
    if primary_site in ["gastrointestinal_tract_(site_indeterminate)", "salivary_gland", "peritoneum"]:
        return "gastrointestinal_system_cancer"

    if primary_site == "upper_aerodigestive_tract":
        if site_subtype_1=="pharynx" or site_subtype_1=="mouth":
            return "gastrointestinal_system_cancer"
        return "respiratory_system_cancer"
    
    if primary_site == "skin":
        return "integumentary_system_cancer"
    
    if primary_site == "breast":
        if histology_subtype_1 == "ductal_carcinoma":
            return "breast_ductal_carcinoma"
        return "breast_cancer"
    
    if primary_site == "ovary":
        return "ovarian_cancer"
    
    if primary_site == "endometrium":
        return "endometrial_cancer"
    
    if primary_site == "cervix":
        return "cervical_cancer"
    
    if primary_site == "fallopian_tube":
        return "female_reproductive_organ_cancer"
    
    if primary_site =="prostate":
        if histology_subtype_1 == "adenocarcinoma":
            return "prostate_adenocarcinoma"
        return "male_reproductive_organ_cancer"
        
    if primary_site== "kidney":
        if histology_subtype_1 == "clear_cell_renal_cell_carcinoma":
            return "clear_cell_renal_cell_carcinoma"
        return "kidney_cancer"
    
    if primary_site == "urinary_tract":
        return "urinary_bladder_cancer"
    
    if primary_site == "pancreas":
        if histology_subtype_1=="ductal_carcinoma":
            return "pancreatic_ductal_carcinoma"
        return "pancreatic_cancer"
    
    if primary_site == "thyroid":
        return "thyroid_gland_cancer"
    
    if primary_site == "adrenal_gland":
        return "adrenal_gland_cancer"
    if primary_site == "pituitary" or primary_site == "parathyroid":
        return "endocrine_gland_cancer"
    
    
    if primary_site == "lung":
        if histology_subtype_1 == "adenocarcinoma":
            return "lung_adenocarcinoma"
        if histology_subtype_1 == "squamous_cell_carcinoma":
            return "lung_squamous_cell_carcinoma"
        return "lung_cancer"
    
    
    if primary_site == "central_nervous_system":
        if site_subtype_1 == "brain":
            if primary_histology == "glioma":
                return "brain_glioma"
            return "brain_cancer"
        return "central_nervous_system_cancer"
    if primary_site == "meninges":
        return "central_nervous_system_cancer"
    if primary_site == "autonomic_ganglia":
        return "nervous_system_cancer"
    
        
    if primary_site == "haematopoietic_and_lymphoid_tissue":
        if primary_histology == "haematopoietic_neoplasm":
            if histology_subtype_1 == "acute_myeloid_leukaemia":
                return "acute_myeloid_leukaemia"
            return "leukemia"
        if primary_histology == "lymphoid_neoplasm":
            if histology_subtype_1 == "chronic_lymphocytic_leukaemia-small_lymphocytic_lymphoma":
                return "chronic_lymphocytic_leukemia"
            if histology_subtype_1 == "acute_lymphoblastic_T_cell_leukaemia":
                return "T_cell_acute_lymphoblastic_leukemia"
            if histology_subtype_1 == "diffuse_large_B_cell_lymphoma":
                return "diffuse_large_B_cell_lymphoma"
            if histology_subtype_1 == "acute_lymphoblastic_B_cell_leukaemia":
                return "B_cell_acute_lymphoblastic_leukemia"
#             if histology_subtype_1 in ["acute_lymphoblastic_leukaemia", "adult_T_cell_lymphoma-leukaemia"]:
#                 return "lymphoid_leukemia"
            if "B_cell" in histology_subtype_1 or "T_cell" in histology_subtype_1:
                return "lymphoid_leukemia"
            return "lymphoma"
        return "immune_system_cancer"
    if primary_site == "thymus":
        return "immune_system_cancer"
    if primary_site in ["bone", "pleura"]:
        return "connective_tissue_cancer"

    
    return ""

def map_histological_subtype(primary_site, site_subtype_1, site_subtype_2, site_subtype_3, primary_histology, histology_subtype_1, histology_subtype_2, histology_subtype_3):
    if "carcinoma" in primary_histology:
        if histology_subtype_1 == "squamous_cell_carcinoma":
            return "squamous_cell_carcinoma"
        if histology_subtype_1 == "adenocarcinoma":
            return "adenocarcinoma"
        return "carcinoma"
    if primary_histology == "glioma":
        if "astrocytoma" in histology_subtype_1:
            return "astrocytoma"
        return "malignant_glioma"
    return ""