---
Logging in
- ssh -p 47916 user@206.168.82.14
---
Transferring files from local to cloud
- scp -P 47916 "/Users/administrator/AI/GPT-Valkyrie/Notes/LN - EE F [Research Notes] V1.ipynb" user@206.168.82.14:/home/user/
Check availablity of local path
- ls "/Users/administrator/AI/GPT-Valkyrie/Math Testing"
---
Transferring files from cloud to local
- scp -P 47916 user@206.168.82.14:/home/user/Welcome.ipynb "/Users/administrator/AI/GPT-Valkyrie/Notes/"