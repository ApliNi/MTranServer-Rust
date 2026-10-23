
## 前端

修改前端资源 `src/ui/*`, 需要递增 `src/version.rs` 中的 `UI_VERSION`.

## 翻译质量差异测试

修改完代码后和 git 提交前运行 `cargo run -p mtranserver-test --bin regression`, 然后检查 `test/cases/regression.json` 的 git diff.
