README: Committing and Tagging a New Version of CurrentPrediction
============================================================

This guide explains how to commit and tag a new version of your project, assuming you have finished making updates in z:\CurrentPrediction and need to update your GitHub repository using z:\CurrentPrediction-clean.

1. Review and Prepare
---------------------
- Ensure all your changes in z:\CurrentPrediction are complete and tested.
- Make sure no files over 100 MB are present (especially in dataset/).

2. Copy Files to the Clean Directory
------------------------------------
- Copy all updated files and folders (except large or ignored files) from z:\CurrentPrediction to z:\CurrentPrediction-clean.
- Do NOT copy dataset/ or any files you want to keep out of GitHub.

3. Stage and Commit Changes
---------------------------
Open a terminal and navigate to z:\CurrentPrediction-clean:

    cd z:\CurrentPrediction-clean

Stage all changes:

    git add -A

Commit your changes (edit the message as needed):

    git commit -m "Describe your changes here"

4. Push to GitHub
-----------------
Force-push to update the remote repository:

    git push origin main --force

5. Tag the New Version
----------------------
Tag the latest commit (replace NEW_TAG with your version, e.g., GANv1.1):

    git tag NEW_TAG
    git push origin NEW_TAG

6. Verify
---------
- Check your GitHub repository to ensure the new commit and tag appear as expected.

Notes:
------
- Only use z:\CurrentPrediction-clean for git operations and pushing to GitHub.
- Keep z:\CurrentPrediction as your main working directory for development.
- Always avoid adding files over 100 MB to the clean directory before committing.
- If you need to remove a file from history, repeat the git filter-repo process as before.

For help, see this README or ask for assistance!