# Project Context

- The canonical runtime environment for this project is the user's remote server, not the local Windows machine.
- Some paths in this project may be server/Linux paths rather than local Windows paths.
- Scripts are usually intended to be run as Bash scripts on the server.
- Do not rely on the local Windows environment to validate code behavior for this project.
- When verification is needed, prefer static inspection or explain what should be tested on the server. The user will sync through GitHub and run server-side tests manually.
