# Beginner Guide: Create and Share a Private GitHub Repository

This guide is written for **Ihtisham Ul Haq** and assumes no prior GitHub experience.

## Important first point

A **Private** GitHub repository is visible only to its owner and people who are explicitly granted access.

Sending the private repository link by itself does **not** give another person access. They must be invited through GitHub and accept the invitation.

### Personal account versus organization

- **Personal private repository:** simplest setup. Invited collaborators can read and also push/write changes.
- **Organization private repository:** better when another person should only be able to view the repository. Organization repositories support a **Read** role.

If the recipient is a trusted co-author who may contribute code, a personal private repository is usually sufficient. If the recipient should only inspect the code, use an organization and grant **Read** access.

---

## Option A — Simplest: private repository under your personal account

### Step 1 — Sign in

1. Open GitHub in your browser.
2. Sign in to your GitHub account.

### Step 2 — Create the repository

1. Click the **+** button in the upper-right corner.
2. Click **New repository**.
3. Under **Owner**, select your own GitHub account.
4. Enter a repository name. A suitable example is:

   `Measurement-Aware-UAV-RMAL`

5. Optional description:

   `Reproducible measurement-aware reinforcement learning pipeline for UAV downlink resource control.`

6. Select **Private**.
7. Do **not** select “Add a README file,” because this repository already contains a README.
8. Do **not** add a `.gitignore` from GitHub, because one is already included.
9. Do **not** choose a license unless you intentionally want to license the unpublished code.
10. Click **Create repository**.

### Step 3 — Upload this repository

For a beginner, **GitHub Desktop** is the easiest reliable method.

1. Install GitHub Desktop from the official GitHub Desktop website.
2. Sign in to GitHub Desktop with the same GitHub account.
3. Extract the ZIP supplied with this guide.
4. Open GitHub Desktop.
5. Choose **File > Add local repository**.
6. Select the extracted `Ihtisham_Ul_Haq_GitHub_Repository` folder.
7. If GitHub Desktop says the folder is not yet a Git repository, choose **Create a repository**.
8. Keep the repository name you created on GitHub.
9. Make the first commit with a message such as:

   `Initial research repository`

10. Click **Publish repository** or **Push origin**.
11. Confirm that **Keep this code private** is enabled if GitHub Desktop presents that option.

After the upload finishes, open the repository on GitHub and confirm that the files appear individually. Do not upload only the ZIP as a single file.

### Step 4 — Verify privacy

1. Open the repository on GitHub.
2. Near the repository name, confirm that it shows **Private**.
3. Open **Settings**.
4. Check repository visibility if needed.

### Step 5 — Invite a collaborator

1. Open the private repository.
2. Click **Settings**.
3. In the left sidebar, open **Collaborators** under the access section.
4. Click **Add people**.
5. Enter the person's GitHub username or email address.
6. Select the correct person.
7. Send the invitation.
8. The person must accept the invitation before the repository becomes visible to them.

### Important permission warning

For a repository owned by a **personal GitHub account**, an invited collaborator has write access as well as read access. Use this only for people you trust to contribute to the repository.

---

## Option B — Private and view-only for selected people

Use this option if a professor, reviewer, or external person should be able to inspect the repository without being able to push changes.

### Step 1 — Create a GitHub organization

1. Click your profile picture on GitHub.
2. Open **Your organizations**.
3. Choose **New organization**.
4. Follow GitHub's setup steps.

Choose an organization name appropriate for your research work.

### Step 2 — Create the repository inside the organization

1. Create a new repository.
2. Choose the organization as **Owner**.
3. Set repository visibility to **Private**.
4. Upload/push the cleaned repository files.

### Step 3 — Grant view-only access

1. Open the organization repository.
2. Go to **Settings**.
3. Open the repository access/people section.
4. Add the required person.
5. Assign the **Read** role.

The Read role lets the person view and pull the repository without normal write access.

---

## Recommended setup for unpublished academic work

For unpublished research code, keep the repository **Private** until publication or until you intentionally decide to make the code public.

Before inviting anyone, confirm that the repository does not contain:

- passwords;
- API keys;
- authentication tokens;
- private certificates;
- confidential reviewer material;
- personal identifiers that are not needed;
- copyrighted datasets that you do not have permission to redistribute.

The supplied `.gitignore` helps prevent common local files and secrets from being committed, but you should still inspect new files before each upload.

---

## How to update the repository later with GitHub Desktop

Whenever you change your code:

1. Save the changed files in the local repository folder.
2. Open GitHub Desktop.
3. Review the changed files shown on the left.
4. Enter a short commit message, for example:

   `Update uncertainty analysis`

5. Click **Commit to main**.
6. Click **Push origin**.

The private GitHub repository will then contain your latest version.

---

## How to remove someone's access later

1. Open the repository on GitHub.
2. Go to **Settings**.
3. Open **Collaborators** or the repository access section.
4. Find the person.
5. Remove their access.

Their GitHub access to the private repository will be revoked.

---

## Ownership in this cleaned repository

The cleaned repository explicitly identifies:

- **Owner:** Ihtisham Ul Haq
- **Creator:** Ihtisham Ul Haq
- **Maintainer:** Ihtisham Ul Haq

This information is present in the README, `AUTHORS.md`, Python source metadata, citation metadata, and Excel workbook metadata.
