# Many-to-Many Matching of Building Footprints Using Generalization Hierarchies

## Author
Nikola Grlj, MSc

## Supervisor
Prof. Dr. Edzer Pebesma, Institute for Geoinformatics, Munster University<br>
MSc Annika Bonerath, Institute for Geodesy and Geoinformatics, Bonn University<br>

## Other contributors
MSc Alexander Naumann, Institute for Geodesy and Geoinformatics, Bonn University<br>
Prof. Dr. Jan-Henrik Haunert, Institute for Geodesy and Geoinformatics, Bonn University<br>
MSc Peter Rottman, Institute for Geodesy and Geoinformatics, Bonn University<br>
Prof. Dr. Stefan Canzar, Heidelberg University<br>

## Institution
Munster University<br>
Bonn University<br>

## Abstract
Thesis is outlining newly developed algorithm for m:n polygon matching using generalization hierarchies and linear programming.
From two datasets representing same spatial extent, a generalization is performed on each and as such used to build a rooted tree. Two trees are joined into
a bipartite graph for which weighted edges are created from Jaccard Index of the underlying nodes polygons.
Bipartite graph is solved in two ways; using Integer Linear Programming and using approximation algorithm. Results show possible m:n matches as well as comparative but faster results using approximation algorithm.

## Table of Contents
- [Introduction](#introduction)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The topic builds on top of previous research done by Geoinformation group from Bonn University (members mentioned above). It is common occurence that for the same spatial extent more than one map is available. Maps represent datasets, a collection of polygons with respective vertex coordinates representing building or land division. Often there will exist a mismatch between two maps. This thesis is trying to implement an algorithm which would detect corresponding polygons between the two datasets.
## Methodology
Detection of corresponding polygons or matching of polygons can come in several different shapes. If there is one polygon from one dataset matched to one polygon from another dataset, we call this a 1:1 match. If more polygons are included, we can have 1:n, m:1 or m:n match. Previous research by done in the group focused on m:1 and 1:n matching, and m:n matching where the latter was done with all possible legal polygon groups in mind. This thesis instead tries to build a generalization hierarchy from each dataset into a rooted tree, and then find best matching groups which provide highest value of the objective function. Objective function represents sum of weights of chosen matches. To maximize this, method of Integer Linear Programming is employed. This method becomes increasingly expensive for large datasets, hence an alternative solution is proposed as a comparison for providing approximate result of matching process. In the end, these two results are compared to unrestricted grouping method.
## Results
Results are satisfactory. The algorithm has ability to detect m:n matches. Using approximation method significantly reduce the computation time with resulting objective function almost always equal to the results obtained with ILP. Matching using generalization hierarchies does provides less optimal results from the unrestrained matching. It is important to mention that the weight used here is Intersection over Union (IoU) which can be replaced by other appropriate weighting method.
## Installation
It is possible to set up the code to run either locally or in Google Colab environment. Running times will depend as Google Colab has a runtime limit and will break for larger datasets. 
Some important caveats are the following: Gurobi librari is non-standard and has to be downloaded from Gurobi directly with licence. Depending on what kind of dataset you have there are different steps which have to be followed. Workflow is described for the full process when starting from ground up.  
### Step 1 - Preprocessing  
User should have two datasets of polygons representing same spatial extent. The datasets must be a shape file. Each of the two datasets has to be generalized with a Delaunay triangulation process and this is done by using generalization.jar file (Java must be installed on your computer). Folder "testing/data" should contain another folder with the name of your dataset. That folder has to contain shape files. One folder per dataset. When folders are redy with shape files inside, a python script called "runAllDatasets.py" should be opened, lines 30 and 32 adjusted in order to be able to locate shape files, and then the script should be ran. If everything goes well, folder "tri" will contain new folders named same like your folders which are holding shape files. Inside will be files used to create generalization hierarchy of your dataset.  
### Step 2 - Building trees
Now you should have two folders (one for each dataset) in the "tri" folder. To build the trees from these two folders, navigate to script "my_methods". This script will create two trees from two datasets (set correct path in lines 10 and 11) and combine these two trees into a bipartite graph with weighted edges between vertices of tree 1 and tree 2. The two trees and bipartite graph are saved for further use, as well as png files representing the trees.  
### Step 3 - Optimization
This is the final step. In the beginning of the code, set up paths for previously saved trees and bipartite graph. In the end of the script, it is possible to adjust commented blocks of the code in order to run one pair of datasets or a batch set. Batch running is appropriate for multiple dataset pairs and lambda values (constant removed from Jaccard Index weights). When optimization has been ran, there will be verbatim output in the console, as well as some files which will be saved. Files are saved in "optimizations" folder under respective names and with the appropriate timestamp. Enter folder to find shape files used and see the matching data for both Canzar and ILP algorithm. "report.txt" file contains most important output for the optimization.
### Appendix
Folder "temp_test_file" functions as a holder for whichever shape files you want to be using.  
Folder hierarchy below shows the most important folders. In reality, all the intermediate files which have been created during the development process are left.


```bash
# Clone the repository
git clone https://github.com/greymouse1/hierarchy_generator.git

# Navigate into the repository
cd hierarchy_generator

# Install dependencies
pip install -r requirements.txt
```
```
hierarchy_generator/
├── README.md
├── T1.pkl
├── T1_auerberg.pkl
├── T1_auerberg_atkis_3m.pkl
├── T1_auerberg_atkis_alex.pkl
├── T1_berkum_atkis.pkl
├── T1_beuel_ost_atkis_01.pkl
├── T1_dottendorf_atkis_01.pkl
├── T1_dransdorf_atkis_alex.pkl
├── T1_dransdorf_atkis_gen.pkl
├── T1_endenich_atkis_alex.pkl
├── T1_endenich_atkis_cut.pkl
├── T1_kottenforst_atkis_gen.pkl
├── T1_kottenforst_atkis_gen.pkl.zip
├── T1_topography.png
├── T1_werthhoven_atkis.pkl
├── T1_zentrum_atkis_alex.pkl
├── T2.pkl
├── T2_auerberg.pkl
├── T2_auerberg_osm_3m.pkl
├── T2_auerberg_osm_alex.pkl
├── T2_berkum_osm.pkl
├── T2_beuel_ost_osm_01.pkl
├── T2_dottendorf_osm_01.pkl
├── T2_dransdorf_osm_alex.pkl
├── T2_dransdorf_osm_gen.pkl
├── T2_endenich_osm_alex.pkl
├── T2_endenich_osm_cut.pkl
├── T2_kottenforst_osm_gen.pkl
├── T2_kottenforst_osm_gen.pkl.zip
├── T2_topography.png
├── T2_werthhoven_osm.pkl
├── T2_zentrum_osm_alex.pkl
├── auerberg_20240701_083910
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── auerberg_alex_0.1_20240711_162621
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── auerberg_alex_0.2_20240711_162902
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── auerberg_alex_0.3_20240711_163141
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── auerberg_alex_0_20240711_162317
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── auerberg_alex_20240709_164221
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── auerberg_alex_20240727_122737
│   └── report.txt
├── auerberg_alex_20240727_123830
│   └── report.txt
├── auerberg_alex_20240727_125801
├── auerberg_alex_20240727_131448
├── auerberg_atkis_wkt_polygons.txt
├── auerberg_atkis_wkt_polygons_alex.txt
├── auerberg_osm_wkt_polygons.txt
├── auerberg_osm_wkt_polygons_alex.txt
├── batch
│   ├── auerberg_alex
│   ├── berkum
│   ├── dransdorf_alex
│   ├── endenich_alex
│   └── werthhoven
├── berkum_0.1_20240711_170002
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── berkum_0.2_20240711_170838
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── berkum_0.3_20240711_171701
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── berkum_0_20240711_164831
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── berkum_20240709_145017
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── berkum_alex_lambda0.3, _20240711_135303
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── berkum_alex_lambda0.4, _20240711_141253
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── berkum_atkis_wkt_polygons.txt
├── berkum_osm_wkt_polygons.txt
├── beuel-ost-0.1_20240701_085757
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── dataset.py
├── dottendorf_0.1_20240701_161512
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── dransdorf_20240629_102147
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── dransdorf_alex_0.1_20240711_194914
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── dransdorf_alex_0.2_20240711_195606
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── dransdorf_alex_0.3_20240711_200252
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── dransdorf_alex_0_20240711_194137
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── dransdorf_alex_20240710_002118
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── dransdorf_atkis_gen_wkt_polygons.txt
├── dransdorf_atkis_wkt_polygons.txt
├── dransdorf_osm_gen_wkt_polygons.txt
├── dransdorf_osm_wkt_polygons.txt
├── dummy2_wkt_polygons.txt
├── dummy_20240704_114456
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── dummy_20240721_124214
│   └── report.txt
├── dummy_20240727_121745
│   └── report.txt
├── dummy_20240727_122149
│   └── report.txt
├── dummy_wkt_polygons.txt
├── endenich_alex_0.1_20240711_184815
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── endenich_alex_0.2_20240711_190624
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── endenich_alex_0.3_20240711_192405
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── endenich_alex_0_20240711_182305
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── endenich_alex_20240710_163956
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── endenich_atkis_cut_wkt_polygons.txt
├── endenich_atkis_wkt_polygons.txt
├── endenich_cut_20240708_110226
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── endenich_cut_20240708_150947
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── endenich_cut_20240708_151205
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── endenich_osm_cut_wkt_polygons.txt
├── endenich_osm_wkt_polygons.txt
├── folder_tree.txt
├── generalization.jar
├── graph_data.csv
├── jaccard_index.graphml
├── jaccard_index_auerberg.graphml
├── jaccard_index_auerberg_3m.graphml
├── jaccard_index_auerberg_alex.graphml
├── jaccard_index_berkum.graphml
├── jaccard_index_beuel_ost_01.graphml
├── jaccard_index_dottendorf_01.graphml
├── jaccard_index_dransdorf_alex.graphml
├── jaccard_index_dransdorf_gen.graphml
├── jaccard_index_endenich_alex.graphml
├── jaccard_index_endenich_cut.graphml
├── jaccard_index_kottenforst_gen.graphml
├── jaccard_index_kottenforst_gen.graphml.zip
├── jaccard_index_werthhoven.graphml
├── jaccard_index_zentrum_alex.graphml
├── json_reader.py
├── kottenforst_east_atkis_0.000100_wkt_polygons.txt
├── kottenforst_east_osm_0.000010_wkt_polygons.txt
├── matches_CANZAR.txt
├── matches_ILP.txt
├── mprofile_20240708151546.dat
├── mprofile_20240708155705.dat
├── my_methods.py
├── newly_generalized
│   ├── auerberg_generalization
│   ├── berkum_generalization
│   ├── dransdorf_generalization
│   ├── endenich_generalization
│   ├── werthhoven_generalization
│   └── zentrum_generalization
├── optimization.py
├── optimizations
│   ├── auerberg_alex_20240727_122737
│   ├── auerberg_alex_20240727_123830
│   ├── auerberg_alex_20240727_125801
│   ├── auerberg_alex_20240727_131448
│   ├── dummy_20240721_124214
│   ├── dummy_20240727_121745
│   └── dummy_20240727_122149
├── output.log
├── plot.py
├── requirements.txt
├── runAllDatasets.py
├── temp_test_files
│   ├── auerberg_atkis_02
│   ├── auerberg_atkis_gen
│   ├── auerberg_atkis_in_progress
│   ├── auerberg_atkis_trim
│   ├── auerberg_osm_02
│   ├── auerberg_osm_gen
│   ├── auerberg_osm_trim
│   ├── beuel_ost_atkis_01
│   ├── beuel_ost_osm_01
│   ├── dottendorf_atkis_01
│   ├── dottendorf_osm_01
│   ├── dransdorf_atkis_gen
│   ├── dransdorf_osm_gen
│   ├── duisdorf_atkis_02
│   ├── duisdorf_osm_02
│   ├── dummy
│   ├── dummy2
│   ├── endenich_atkis_02
│   ├── endenich_atkis_cut
│   ├── endenich_osm_02
│   ├── endenich_osm_cut
│   ├── kessenich_atkis_01
│   ├── kessenich_osm_01
│   ├── roettgen_atkis_01
│   ├── roettgen_atkis_gen
│   ├── roettgen_osm_01
│   ├── suetstadt_osm_gen
│   ├── zentrum_atkis
│   ├── zentrum_osm
│   └── zentrum_osm_gen
├── testing
│   └── data
├── tree_generator.py
├── tree_polygons.cpg
├── tree_polygons.dbf
├── tree_polygons.shp
├── tree_polygons.shx
├── trees
│   └── dummy_20240721_130424
├── tri
│   ├── auerberg_atkis_3m
│   ├── auerberg_atkis_gen
│   ├── auerberg_atkis_in_progress
│   ├── auerberg_osm_3m
│   ├── auerberg_osm_gen
│   ├── dransdorf_atkis_gen
│   ├── dransdorf_osm_gen
│   ├── duisdorf_atkis_02
│   ├── duisdorf_osm_02
│   ├── dummy
│   ├── dummy2
│   ├── endenich_atkis_cut
│   ├── endenich_osm_cut
│   ├── kottenforst_east_atkis_0.000100
│   ├── kottenforst_east_osm_0.000010
│   ├── roettgen_atkis_gen
│   └── suetstadt_osm_gen
├── utils.py
├── werthhoven_0.1_20240711_161628
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── werthhoven_0.2_20240711_161758
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── werthhoven_0.3_20240711_161928
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── werthhoven_0_20240711_161453
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── werthovven_20240709_152303
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── zentrum_alex_20240711_112844
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── zentrum_alex_lambda0.1, _20240711_113204
│   ├── matches_ILP.json
│   ├── matches_canzar.json
│   ├── report.txt
│   ├── shp_files_CANZAR
│   └── shp_files_ILP
├── zentrum_atkis_wkt_polygons.txt
└── zentrum_osm_wkt_polygons.txt
```
