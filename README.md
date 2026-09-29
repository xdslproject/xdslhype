# xdsl-template

A template for projects using xDSL.

## Usage

### Create a new project

Use [sync-template](https://github.com/XXCC-Unicorn-Press/sync-template) to create a project from this template:

```bash
uvx sync-template init gh:xdslproject/xdsl-template my-new-project
cd my-new-project
```

To add the template remote to an existing repository instead:

```bash
cd my-existing-repo
uvx sync-template init gh:xdslproject/xdsl-template --existing
```

### Customize the project

1. Pick a name for your project
2. Fill in the blanks:
   - Rename `src/xdsltemplate` to `src/projectname`
   - Find all instances of "TODO: " in the project and complete them

     | File | Item to replace | Content |
     | ---- | --------------- | ------- |
     | `mkdocs.yml` | `site_name` | Documentation website title |
     | `mkdocs.yml` | `site_author` | Documentation website authors |
     | `mkdocs.yml` | `copyright` | Documentation website copyright |
     | `mkdocs.yml` | `repo_name` | Project name |
     | `mkdocs.yml` | `repo_url` | Project repository URL |
     | `mkdocs.yml` | `extra.social.link` | Project repository/organisation URL |
     | `pyproject.toml` | `project.name` | Project name |
     | `pyproject.toml` | `project.description` | Project description |
     | `pyproject.toml` | `project.authors` | Project authors |
     | `pyproject.toml` | `project.urls.Homepage` | Project homepage URL |
     | `pyproject.toml` | `project.urls.Source Code` | Project repository URL |
     | `pyproject.toml` | `project.urls.Issue Tracker` | Project issue tracker URL |
     | `docs.yml` | `on` | Uncomment to automatically deploy documentation site |
     | `src/xdsltemplate/__init__.py` | `name` | Replace with name of the project |
     | `src/xdsltemplate/` | Directory name | Replace with name of the project |

3. Run `make install` to set up your environment

### Sync template updates

When the template has updates, sync them into your project:

```bash
make sync
```

This runs `uvx sync-template sync`.

## Example project structure

```text
.
├── docs/
│   ├── index.md              # Hand-written documentation splash page
│   └── reference.md          # Auto-generated source code documentation
├── src/
│   └── xdsltemplate/         # Project source code:
│       ├── dialects/         # - xDSL dialect implementations
│       ├── frontend/         # - Parsing and lexing implementations
│       ├── interpreters/     # - Interpreter implementations
│       └── transforms/       # - Rewriting transformation implementations
└── tests/                    # All tests, both pytest and filecheck
    └── filecheck/            # filecheck .mlir files
```

## Developer features for free

- Developer tools including auto-formatting, linting, and type-checking as both
  pre-commit hooks and GitHub Actions.
- Testing through both pytest and LLVM filecheck
- Documentation website, optionally deployed to GitHub pages
