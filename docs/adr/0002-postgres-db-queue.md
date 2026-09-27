# ADR-0002：使用 PostgreSQL 作为元数据存储与初始作业队列

状态：已接受

## 决定

API 将作业、事件、资产、工具运行和报告写入 PostgreSQL。Ingest Worker 使用 `FOR UPDATE SKIP LOCKED` 原子抢占 `queued` 作业。

## 原因

- 首阶段避免额外引入 Redis/RabbitMQ；
- 作业状态与业务元数据在同一事务系统中，恢复和审计更直接；
- `SKIP LOCKED` 支持多个 worker 并发且不会重复领取同一作业。

## 限制

高吞吐或复杂优先级/延迟队列出现后，再评估 Temporal、Kafka 或专用消息队列。worker 必须保持幂等，数据库队列不能替代业务级去重。
