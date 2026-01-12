# Please follow

# Claude Code Guidelines for LalaSearch

## Most Important Rule

**ASK QUESTIONS FIRST** - Don't blindly execute requests!

- Think critically about the best solution
- Ask when you see a better approach
- Question potential conflicts with existing code
- Clarify unclear requirements
- Discuss trade-offs before implementing

You're a development partner, not just a code executor. Your expertise matters!

## Quick Reference

### TDD Cycle
1. **Analyze**: Identify corner cases and requirements
2. **Red**: Write failing tests
3. **Green**: Minimal implementation to pass tests
4. **Refactor**: Improve code quality (optional)

### Before Every Commit
```bash
./scripts/pre-commit.sh
```

### Completing Features

**Every feature MUST be completed with**:
1. Run `./scripts/pre-commit.sh`
2. Commit: `git add . && git commit -m "feat: description"`

**Never consider a feature complete until it is committed!**

## Project Structure

- `docs/` - All documentation
- `scripts/` - Development tools

## Key Principles

1. Tests before code
2. High code quality (zero clippy warnings)
3. Proper formatting (rustfmt)
4. Complete features with commits
5. Document architectural decisions

## Best Practices

1. **Test Pyramid:** More unit tests, fewer integration tests
2. **Isolation:** Each test should be independent
3. **Fast Feedback:** Keep quick tests fast (<1s per test)
4. **Clear Assertions:** Use descriptive assertion messages
5. **Test Data:** Use realistic sample PDFs
6. **Continuous Testing:** Run tests on every commit
7. **Coverage Goals:** Aim for >80% code coverage on critical paths
8. **Regression Tests:** Add test for every bug fix
