Description: Software package for achieving data reduction in feature space or in particle representation size for modeling datasets as image-varifold particle representations. 

Algorithm Components:
1) Mutual Information Score for Ranking Features by Spatial Variability
   - mutual information computed for distributions of features with respect to vertical and horizontal divisions (approxCode/mrnaMI.py)
   - total mutual information score computed as sum of scores for vertical and horizontal divisions and across 2D stacks in data set (bashScripts/runMI.sh)
2) Optimized Aggregation for Reducing Size of Particle Set Representing Image-Varifold Object
     - high resolution data loaded and reduced representation initialized via stratified subsampling (examples in varap/io/load*.py)
     - reduced representation optimized over choice of mass weights, probability distributions, and physical locations so as to minimize varifold normed difference to high resolution representation (examples in pythonScripts/main_approx*.py)
       


Authors:
Kaitlin Stouffer (kstouff4@jhmi.edu)
Alain Trouvé
Benjamin Charlier
