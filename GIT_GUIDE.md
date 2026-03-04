# Git 使用指南

## 📌 仓库信息

- **仓库地址**: https://github.com/shibuzhuan123/forgery-detection-competition
- **代理设置**: 已配置使用 Miomo 代理 (127.0.0.1:7890)

## 🔄 常用命令

### 查看状态
```bash
git status
```

### 提交修改
```bash
# 添加所有修改
git add .

# 提交修改
git commit -m "描述您的修改内容"

# 推送到远程仓库
git push
```

### 查看提交历史
```bash
git log --oneline --all
```

### 创建新分支
```bash
# 创建并切换到新分支
git checkout -b feature-branch-name

# 推送新分支到远程
git push -u origin feature-branch-name
```

### 合并分支
```bash
# 切换到 main 分支
git checkout main

# 合并其他分支
git merge feature-branch-name

# 推送合并结果
git push
```

## 📝 提交信息规范

好的提交信息应该清晰描述修改内容：

```
✅ 好的例子：
- "Add validation set split (10% of training data)"
- "Implement weighted sampler for class imbalance"
- "Fix data augmentation pipeline"

❌ 不好的例子：
- "update"
- "fix bug"
- "tmp"
```

## 🏷️ 版本管理建议

### 开发新功能时
```bash
# 1. 创建新分支
git checkout -b dev/feature-name

# 2. 开发并提交
git add .
git commit -m "Add new feature"

# 3. 推送到远程
git push -u origin dev/feature-name
```

### 修复Bug时
```bash
# 1. 创建修复分支
git checkout -b fix/bug-description

# 2. 修复并提交
git add .
git commit -m "Fix: bug description"

# 3. 推送
git push -u origin fix/bug-description
```

## 🔐 关于凭据

由于安全原因，Git token 凭据文件已清理。
如果将来需要重新配置，请使用：

```bash
git config credential.helper store
git push
# 然后输入用户名和 token
```

## 📊 当前分支结构

```
main (主分支)
├── Initial commit: Baseline training code
└── Add GitHub setup script
```

## 🚀 下一步

现在可以安全地开发新功能了！每次重要修改都记得：
1. `git add` 添加文件
2. `git commit` 提交修改
3. `git push` 推送到 GitHub
