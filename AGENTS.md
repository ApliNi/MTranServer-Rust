## 前端

修改前端资源 `src/ui/*`, 需要递增 `src/version.rs` 中的 `UI_VERSION`.

## 翻译质量差异测试

修改完代码后和 git 提交前运行 `cargo run -p mtranserver-test --bin regression`, 然后检查 `test/cases/regression.json` 的 git diff.

## 通用性设计规范

不可编写不通用/单点过拟合或特例打补丁的代码:
- 不可针对特定案例硬编码修补
- 不可用硬编码词表模拟 NLP 任务
- 排版与预后处理保持泛化与语义完整
