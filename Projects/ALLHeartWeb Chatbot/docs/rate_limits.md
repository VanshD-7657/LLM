---
title: Rate limits
category: Reference
source_url: https://whoisdatacenter.com/api-docs/#rate-limits
---

# Rate limits

-R-e-f-e-r-e-n-c-e-
-
-L-i-m-i-t-s- -a-r-e- -e-n-f-o-r-c-e-d- -p-e-r- -A-P-I- -k-e-y- -a-c-r-o-s-s- -t-h-r-e-e- -r-o-l-l-i-n-g- -w-i-n-d-o-w-s- -(-p-e-r- -m-i-n-u-t-e-,- -p-e-r- -h-o-u-r-,- -p-e-r- -d-a-y-)- -a-n-d- -c-o-n-c-u-r-r-e-n-t- -i-n---f-l-i-g-h-t- -r-e-q-u-e-s-t-s-.- -E-x-c-e-e-d-i-n-g- -a-n-y- -w-i-n-d-o-w- -r-e-t-u-r-n-s- -`-4-2-9- -T-o-o- -M-a-n-y- -R-e-q-u-e-s-t-s-`-.-
-
-#-#-#- -P-l-a-n- -l-i-m-i-t-s-
-
-|- -P-l-a-n- -|- -R-e-q- -/- -m-i-n- -|- -R-e-q- -/- -h-o-u-r- -|- -R-e-q- -/- -d-a-y- -|- -C-o-n-c-u-r-r-e-n-t- -|- -N-o-t-e-s- -|-
-|- ------- -|- ------- -|- ------- -|- ------- -|- ------- -|- ------- -|-
-|- -*-*-F-r-e-e- -T-r-i-a-l-*-*- -|- -3-0- -|- -5-0-0- -|- -5-,-0-0-0- -|- -2- -|- -G-o-o-d- -f-o-r- -t-e-s-t-i-n-g- -A-P-I-s- -|-
-|- -*-*-S-t-a-r-t-e-r-*-*- -|- -6-0- -|- -5-,-0-0-0- -|- -5-0-,-0-0-0- -|- -5- -|- -S-m-a-l-l- -c-o-m-p-a-n-i-e-s- -|-
-|- -*-*-P-r-o-*-*- -|- -1-8-0- -|- -2-0-,-0-0-0- -|- -2-5-0-,-0-0-0- -|- -1-5- -|- -H-e-a-v-y- -A-P-I- -u-s-e-r-s- -|-
-|- -*-*-B-u-s-i-n-e-s-s-*-*- -|- -5-0-0- -|- -1-0-0-,-0-0-0- -|- -1-,-0-0-0-,-0-0-0- -|- -3-0- -|- -L-a-r-g-e- -u-s-a-g-e- -|-
-|- -*-*-E-n-t-e-r-p-r-i-s-e-*-*- -|- -C-u-s-t-o-m- -|- -C-u-s-t-o-m- -|- -C-u-s-t-o-m- -|- -C-u-s-t-o-m- -|- -D-e-d-i-c-a-t-e-d- -l-i-m-i-t-s- -|-
-
-#-#-#- -R-e-s-p-o-n-s-e- -h-e-a-d-e-r-s-
-
-E-v-e-r-y- -r-e-s-p-o-n-s-e- -i-n-c-l-u-d-e-s- -r-a-t-e---l-i-m-i-t- -h-e-a-d-e-r-s-:-
-
-`-`-`-
-X---R-a-t-e-L-i-m-i-t---L-i-m-i-t-:- - - - - - -5-0-0-0- - - - - - - - - -#- -C-a-p- -f-o-r- -c-u-r-r-e-n-t- -w-i-n-d-o-w-
-X---R-a-t-e-L-i-m-i-t---R-e-m-a-i-n-i-n-g-:- - -3-2-0-0- - - - - - - - - -#- -C-a-l-l-s- -l-e-f-t- -b-e-f-o-r-e- -t-h-r-o-t-t-l-e-
-X---R-a-t-e-L-i-m-i-t---R-e-s-e-t-:- - - - - - -1-7-1-3-8-9-0-0-0-0- - - -#- -U-n-i-x- -t-s- -w-h-e-n- -w-i-n-d-o-w- -r-e-s-e-t-s-
-`-`-`-