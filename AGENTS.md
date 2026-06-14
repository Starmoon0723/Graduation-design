# Project Context

- The canonical runtime environment for this project is the user's remote server, not the local Windows machine.
- Some paths in this project may be server/Linux paths rather than local Windows paths.
- Scripts are usually intended to be run as Bash scripts on the server.
- Do not rely on the local Windows environment to validate code behavior for this project.
- When verification is needed, prefer static inspection or explain what should be tested on the server. The user will sync through GitHub and run server-side tests manually.
- 服务器根目录位置在 /XYFS01/HDD_POOL/hitsz_mszhang/hitsz_mszhang_1/MRC/MRC/MRC_project/others/AAA/vlm/Graduation-design， 在运行服务器的bash脚本进行模型训练等等之前，我通常会运行 source /XYFS01//XYFS01/HDD_POOL/hitsz_mszhang/hitsz_mszhang_1/MRC/MRC/MRC_project/others/AAA/vlm/cache_env_new.sh, 该脚本中包括了一些环境路径配置