# Omni-Recall: Neural Knowledge & Long-Term Context Engine

> **Elevate your AI Agent with persistent neural memory.**

Omni-Recall is a standardized Agent Skill for Trae, Cursor, and other AI coding assistants. It provides a robust, vector-based long-term memory system using Supabase (PostgreSQL + pgvector) and APIYI (Embeddings).

## 🌟 Features

- **Cross-Session Memory**: Never lose context between chat restarts.
- **Neural Embeddings**: High-dimensional vector storage for semantic search.
- **Automated Sync**: Easily integrate memory synchronization into your agent's workflow.
- **Enterprise Ready**: Built on top of PostgreSQL and standardized Agent Skill specifications.

## 📦 Installation

To install **Omni-Recall** in your current project, run the following command in your terminal:

```bash
npx skills add https://github.com/[YOUR_GITHUB_USERNAME]/omni-recall --skill omni-recall
```

*Note: Replace `[YOUR_GITHUB_USERNAME]` with your actual GitHub username after pushing the repository.*

## ⚙️ Configuration

1. **Database Setup**: Run the SQL script found in `SKILL.md` in your Supabase SQL editor.
2. **Environment Variables**:
   - `APIYI_TOKEN`: Your API key from [apiyi.com](https://api.apiyi.com).
   - `SUPABASE_PASSWORD`: Your database password.

## 📖 Usage

Once installed, your agent will have access to the instructions in `SKILL.md`. You can also use the CLI directly:

```bash
# Sync current state
python3 omni-recall/scripts/omni_ops.py sync "User prefers Tailwind CSS for frontend."

# Fetch recent context
python3 omni-recall/scripts/omni_ops.py fetch 7
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT
