# Raise Feature Store - Product Requirements Document

**Version:** 1.1
**Status:** Ready for Engineering Review
**Last Updated:** 2026-02-21

> **v1.1 Changes:** Added ETL/Transformations (Section 15) and Airflow Integration (Section 16)

---

## Executive Summary

Raise is a Feature Store API designed for AI researchers working in notebook environments. It provides a Python-first interface for managing ML features in columnar storage backends, with support for derived features, lineage tracking, access control, audit logging, and comprehensive analytics.

This PRD defines the backend systems and middleware required to support the Raise API specification.

---

## Table of Contents

1. [Goals and Non-Goals](#1-goals-and-non-goals)
2. [System Architecture](#2-system-architecture)
3. [Data Model](#3-data-model)
4. [Core Feature Management](#4-core-feature-management)
5. [Derived Features and Expression Engine](#5-derived-features-and-expression-engine)
6. [Versioning System](#6-versioning-system)
7. [Lineage Tracking](#7-lineage-tracking)
8. [Access Control System](#8-access-control-system)
9. [Cross-Organization Access](#9-cross-organization-access)
10. [Audit Logging](#10-audit-logging)
11. [Analytics Engine](#11-analytics-engine)
12. [Live Tables (CDC)](#12-live-tables-cdc)
13. [Dashboards](#13-dashboards)
14. [Alerting System](#14-alerting-system)
15. [ETL and Transformations](#15-etl-and-transformations)
16. [Airflow Integration](#16-airflow-integration)
17. [Storage Requirements](#17-storage-requirements)
18. [API Layer](#18-api-layer)
19. [Non-Functional Requirements](#19-non-functional-requirements)
20. [Appendix: Data Type Specifications](#appendix-a-data-type-specifications)
21. [Appendix: SQL Function Support](#appendix-b-sql-function-support)

---

## 1. Goals and Non-Goals

### Goals

1. **Notebook-First Experience**: Provide intuitive Python APIs callable from Jupyter notebooks
2. **Feature Abstraction**: Create a logical feature layer over physical columnar storage
3. **Derived Features**: Support SQL-like expressions for computed features with automatic lineage
4. **Multi-Tenant Isolation**: Support organization, domain, and project boundaries
5. **Cross-Org Sharing**: Enable controlled feature sharing across organizations
6. **Immutable Versioning**: Ensure schema changes create new versions (audit trail)
7. **Comprehensive Analytics**: Provide aggregations, distributions, statistical tests, and dashboards
8. **Real-Time Updates**: Support CDC-based live tables for analytics materialization
9. **ETL/Transformations**: Support data transformation pipelines with SQL and Python, integrated with Airflow
10. **Incremental Processing**: Checkpoint-based incremental updates with support for late-arriving data

### Non-Goals (v1)

1. Feature serving at inference time (online store)
2. Real-time streaming ingestion
3. Model training integration
4. Feature monitoring/observability dashboards (built-in UI)
5. Backfill orchestration (manual backfills supported, automated orchestration deferred)

---

## 2. System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Client Layer                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Python    │  │   REST API  │  │   gRPC API  │  │   SDK Gen   │    │
│  │   SDK       │  │   Gateway   │  │   Gateway   │  │  (future)   │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          API Gateway / Auth                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   AuthN     │  │   AuthZ     │  │ Rate Limit  │  │   Routing   │    │
│  │   (OAuth2)  │  │   (ACL)     │  │             │  │             │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Service Layer                                    │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐               │
│  │   Feature     │  │   Lineage     │  │   Analytics   │               │
│  │   Service     │  │   Service     │  │   Service     │               │
│  └───────────────┘  └───────────────┘  └───────────────┘               │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐               │
│  │   ACL         │  │   Audit       │  │   Dashboard   │               │
│  │   Service     │  │   Service     │  │   Service     │               │
│  └───────────────┘  └───────────────┘  └───────────────┘               │
│  ┌───────────────┐  ┌───────────────┐                                   │
│  │   Alert       │  │   Live Table  │                                   │
│  │   Service     │  │   Service     │                                   │
│  └───────────────┘  └───────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Compute Layer                                     │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐               │
│  │   Query       │  │   Expression  │  │   Analytics   │               │
│  │   Engine      │  │   Evaluator   │  │   Engine      │               │
│  │   (SQL)       │  │   (Derived)   │  │   (Stats)     │               │
│  └───────────────┘  └───────────────┘  └───────────────┘               │
│  ┌───────────────┐  ┌───────────────┐                                   │
│  │   CDC         │  │   Job         │                                   │
│  │   Processor   │  │   Scheduler   │                                   │
│  └───────────────┘  └───────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Storage Layer                                     │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐               │
│  │   Metadata    │  │   Feature     │  │   Analytics   │               │
│  │   Store       │  │   Store       │  │   Cache       │               │
│  │   (Postgres)  │  │   (Columnar)  │  │   (Redis)     │               │
│  └───────────────┘  └───────────────┘  └───────────────┘               │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐               │
│  │   Audit       │  │   Lineage     │  │   Blob        │               │
│  │   Log         │  │   Graph       │  │   Storage     │               │
│  │   (Append)    │  │   (Neo4j)     │  │   (S3)        │               │
│  └───────────────┘  └───────────────┘  └───────────────┘               │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility |
|-----------|----------------|
| **Feature Service** | CRUD for features, feature groups, projects, domains, orgs |
| **Lineage Service** | Track dependencies, compute transitive closures |
| **Analytics Service** | Execute analyses, manage results, freshness control |
| **ACL Service** | Resolve permissions, cascading inheritance |
| **Audit Service** | Log all operations, query audit history |
| **Dashboard Service** | Store dashboard definitions, render specs |
| **Alert Service** | Evaluate conditions, trigger notifications |
| **Live Table Service** | Manage CDC subscriptions, refresh materialized views |
| **Query Engine** | Execute SQL against feature data |
| **Expression Evaluator** | Validate and compile derived feature expressions |
| **Analytics Engine** | Compute aggregations, distributions, correlations, statistical tests |
| **CDC Processor** | Consume change events, trigger live table refreshes |
| **Job Scheduler** | Schedule async analyses, alert checks, periodic refreshes |

---

## 3. Data Model

### Entity Relationship Diagram

```
Organization (1) ─────< Domain (N)
     │                     │
     │                     │
     ▼                     ▼
   ACL                  Project (N)
                           │
                           │
                           ▼
                    FeatureGroup (N)
                           │
                    ┌──────┴──────┐
                    │             │
                    ▼             ▼
              Feature (N)   LiveTable (N)
                    │
             ┌──────┼──────┐
             │      │      │
             ▼      ▼      ▼
         Version  Lineage  ACL
```

### Metadata Schema

#### organizations
```sql
CREATE TABLE organizations (
    id UUID PRIMARY KEY,
    name VARCHAR(128) UNIQUE NOT NULL,
    display_name VARCHAR(256),
    owner VARCHAR(256) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    settings JSONB DEFAULT '{}'
);
```

#### domains
```sql
CREATE TABLE domains (
    id UUID PRIMARY KEY,
    org_id UUID NOT NULL REFERENCES organizations(id),
    name VARCHAR(128) NOT NULL,
    description TEXT,
    owner VARCHAR(256),
    tags VARCHAR(64)[] DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE(org_id, name)
);
```

#### projects
```sql
CREATE TABLE projects (
    id UUID PRIMARY KEY,
    domain_id UUID NOT NULL REFERENCES domains(id),
    name VARCHAR(128) NOT NULL,
    description TEXT,
    owner VARCHAR(256),
    tags VARCHAR(64)[] DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE(domain_id, name)
);
```

#### feature_groups
```sql
CREATE TABLE feature_groups (
    id UUID PRIMARY KEY,
    project_id UUID NOT NULL REFERENCES projects(id),
    name VARCHAR(128) NOT NULL,
    description TEXT,
    owner VARCHAR(256),
    tags VARCHAR(64)[] DEFAULT '{}',
    entity_keys VARCHAR(128)[] DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE(project_id, name)
);
```

#### features
```sql
CREATE TABLE features (
    id UUID PRIMARY KEY,
    group_id UUID NOT NULL REFERENCES feature_groups(id),
    name VARCHAR(128) NOT NULL,
    version VARCHAR(32) NOT NULL DEFAULT 'v1',
    dtype VARCHAR(256) NOT NULL,
    description TEXT,
    owner VARCHAR(256),
    tags VARCHAR(64)[] DEFAULT '{}',
    nullable BOOLEAN DEFAULT TRUE,
    default_value JSONB,
    derived_from TEXT,  -- SQL expression if derived
    status VARCHAR(32) DEFAULT 'active',  -- active, deprecated, archived
    deprecation_message TEXT,
    physical_column VARCHAR(256),  -- Mapped storage column
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE(group_id, name, version)
);

CREATE INDEX idx_features_group ON features(group_id);
CREATE INDEX idx_features_status ON features(status);
CREATE INDEX idx_features_tags ON features USING GIN(tags);
```

#### feature_lineage
```sql
CREATE TABLE feature_lineage (
    id UUID PRIMARY KEY,
    feature_id UUID NOT NULL REFERENCES features(id),
    upstream_feature_id UUID NOT NULL REFERENCES features(id),
    expression_context TEXT,  -- The expression segment using this upstream
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE(feature_id, upstream_feature_id)
);

CREATE INDEX idx_lineage_feature ON feature_lineage(feature_id);
CREATE INDEX idx_lineage_upstream ON feature_lineage(upstream_feature_id);
```

#### acls
```sql
CREATE TABLE acls (
    id UUID PRIMARY KEY,
    resource_type VARCHAR(32) NOT NULL,  -- organization, domain, project, feature_group, feature
    resource_id UUID NOT NULL,
    readers VARCHAR(256)[] DEFAULT '{}',
    writers VARCHAR(256)[] DEFAULT '{}',
    admins VARCHAR(256)[] DEFAULT '{}',
    inherit BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE(resource_type, resource_id)
);
```

#### external_grants
```sql
CREATE TABLE external_grants (
    id UUID PRIMARY KEY,
    granting_org_id UUID NOT NULL REFERENCES organizations(id),
    grantee_org VARCHAR(128) NOT NULL,
    resource_type VARCHAR(32) NOT NULL,
    resource_id UUID NOT NULL,
    features VARCHAR(128)[] DEFAULT '{}',  -- Empty means all
    permission VARCHAR(32) NOT NULL,  -- read, write
    granted_by VARCHAR(256) NOT NULL,
    expires_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE(granting_org_id, grantee_org, resource_type, resource_id)
);
```

#### audit_log
```sql
CREATE TABLE audit_log (
    id UUID PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    org_id UUID NOT NULL,
    actor VARCHAR(256) NOT NULL,
    actor_org VARCHAR(128),
    action VARCHAR(64) NOT NULL,
    resource_type VARCHAR(32) NOT NULL,
    resource_id UUID NOT NULL,
    resource_path VARCHAR(512),
    metadata JSONB DEFAULT '{}',
    ip_address INET,
    user_agent VARCHAR(512)
) PARTITION BY RANGE (timestamp);

CREATE INDEX idx_audit_timestamp ON audit_log(timestamp);
CREATE INDEX idx_audit_resource ON audit_log(resource_type, resource_id);
CREATE INDEX idx_audit_actor ON audit_log(actor);
CREATE INDEX idx_audit_action ON audit_log(action);
```

#### live_tables
```sql
CREATE TABLE live_tables (
    id UUID PRIMARY KEY,
    group_id UUID NOT NULL REFERENCES feature_groups(id),
    name VARCHAR(128) NOT NULL,
    description TEXT,
    analysis_spec JSONB NOT NULL,
    refresh_policy JSONB NOT NULL,
    status VARCHAR(32) DEFAULT 'active',
    last_refresh TIMESTAMP,
    row_count BIGINT DEFAULT 0,
    storage_path VARCHAR(512),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE(group_id, name)
);
```

#### live_table_refresh_history
```sql
CREATE TABLE live_table_refresh_history (
    id UUID PRIMARY KEY,
    live_table_id UUID NOT NULL REFERENCES live_tables(id),
    status VARCHAR(32) NOT NULL,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    rows_processed BIGINT,
    trigger VARCHAR(32),  -- cdc, scheduled, manual
    error_message TEXT
);
```

#### dashboards
```sql
CREATE TABLE dashboards (
    id UUID PRIMARY KEY,
    org_id UUID NOT NULL REFERENCES organizations(id),
    name VARCHAR(128) NOT NULL,
    description TEXT,
    charts JSONB DEFAULT '[]',
    parameters JSONB DEFAULT '[]',
    layout JSONB DEFAULT '{}',
    published BOOLEAN DEFAULT FALSE,
    published_url VARCHAR(512),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE(org_id, name)
);
```

#### analytics_alerts
```sql
CREATE TABLE analytics_alerts (
    id UUID PRIMARY KEY,
    org_id UUID NOT NULL REFERENCES organizations(id),
    name VARCHAR(128) NOT NULL,
    analysis_spec JSONB NOT NULL,
    condition_spec JSONB NOT NULL,
    notify VARCHAR(256)[] NOT NULL,
    channels VARCHAR(32)[] DEFAULT '{email}',
    check_interval INTERVAL NOT NULL,
    enabled BOOLEAN DEFAULT TRUE,
    status VARCHAR(32) DEFAULT 'active',
    last_check TIMESTAMP,
    last_triggered TIMESTAMP,
    trigger_count INT DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE(org_id, name)
);
```

#### analysis_results
```sql
CREATE TABLE analysis_results (
    id UUID PRIMARY KEY,
    org_id UUID NOT NULL,
    analysis_spec_hash VARCHAR(64) NOT NULL,
    analysis_spec JSONB NOT NULL,
    result_data JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    freshness_at TIMESTAMP NOT NULL,
    expires_at TIMESTAMP
);

CREATE INDEX idx_results_hash ON analysis_results(analysis_spec_hash);
CREATE INDEX idx_results_freshness ON analysis_results(freshness_at);
```

#### analysis_jobs
```sql
CREATE TABLE analysis_jobs (
    id UUID PRIMARY KEY,
    org_id UUID NOT NULL,
    analysis_spec JSONB NOT NULL,
    status VARCHAR(32) NOT NULL,  -- pending, running, completed, failed, cancelled
    result_id UUID REFERENCES analysis_results(id),
    error_message TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE INDEX idx_jobs_status ON analysis_jobs(status);
```

---

## 4. Core Feature Management

### Requirements

#### 4.1 Feature CRUD Operations

| Operation | Description | Backend Action |
|-----------|-------------|----------------|
| Create Feature | Create new feature with schema | Insert metadata + create physical column |
| Get Feature | Retrieve feature by path | Query metadata |
| Update Feature | Update mutable metadata | Update metadata (no schema changes) |
| Delete Feature | Soft delete feature | Mark status='archived' |
| Deprecate Feature | Mark as deprecated | Set status='deprecated' + message |
| List Features | List with filters | Query with pagination |

#### 4.2 Physical Storage Mapping

Each feature maps to a physical column in the storage backend:

```
Logical: acme/mlplatform/recommendation/user-signals/user_embedding@v1
Physical: acme__mlplatform__recommendation__user_signals.user_embedding_v1
```

**Mapping Rules:**
1. Table name: `{org}__{domain}__{project}__{group}`
2. Column name: `{feature}_{version}`
3. Underscores in names replaced with `_u_`

#### 4.3 Idempotent Creation

Support three modes for `if_exists` parameter:

| Mode | Behavior |
|------|----------|
| `error` | Raise `FeatureExistsError` if exists |
| `skip` | Return existing feature silently |
| `update` | Update mutable fields if exists |

#### 4.4 Bulk Operations

| Operation | Input | Backend Optimization |
|-----------|-------|---------------------|
| `create_features_from_schema` | `{name: dtype}` dict | Batch insert |
| `create_features` | List of feature dicts | Batch insert |
| `create_features_from_file` | YAML/JSON file path | Parse + batch insert |

---

## 5. Derived Features and Expression Engine

### Requirements

#### 5.1 Expression Syntax

Support SQL-like expressions for derived features:

```sql
-- Arithmetic
click_count / NULLIF(impression_count, 0)

-- Aggregations
(raw_score - AVG(raw_score)) / STDDEV(raw_score)

-- Conditionals
CASE WHEN revenue > 10000 THEN 'platinum'
     WHEN revenue > 1000 THEN 'gold'
     ELSE 'bronze' END

-- Vector operations
DOT(user_embedding, item_embedding)
COSINE_SIMILARITY(vec_a, vec_b)

-- Cross-group references
user-signals.click_count / item-signals.impression_count

-- Window functions
SUM(revenue) OVER (PARTITION BY user_id ORDER BY timestamp ROWS 7 PRECEDING)
```

#### 5.2 Supported Functions

| Category | Functions |
|----------|-----------|
| **Aggregations** | `AVG`, `SUM`, `MIN`, `MAX`, `COUNT`, `STDDEV`, `VARIANCE`, `PERCENTILE` |
| **Math** | `ABS`, `CEIL`, `FLOOR`, `ROUND`, `LOG`, `LOG10`, `EXP`, `POWER`, `SQRT`, `SIGN` |
| **Vector** | `DOT`, `COSINE_SIMILARITY`, `L2_DISTANCE`, `L1_DISTANCE`, `NORM`, `NORMALIZE` |
| **String** | `CONCAT`, `LOWER`, `UPPER`, `TRIM`, `LTRIM`, `RTRIM`, `SUBSTRING`, `LENGTH`, `REPLACE` |
| **Conditional** | `CASE WHEN`, `COALESCE`, `NULLIF`, `IF`, `IIF` |
| **Window** | `OVER`, `PARTITION BY`, `ORDER BY`, `ROWS/RANGE`, `LAG`, `LEAD`, `FIRST_VALUE`, `LAST_VALUE` |
| **Null** | `IS NULL`, `IS NOT NULL`, `IFNULL`, `NVL` |

#### 5.3 Expression Validation

The expression evaluator must validate:

1. **Syntax** - Parse SQL expression successfully
2. **References** - All referenced features exist and are accessible
3. **Types** - Operations are type-compatible
4. **Cycles** - No circular dependencies
5. **Permissions** - User has access to all referenced features

**Validation Levels:**

| Level | Behavior |
|-------|----------|
| `strict` | Errors + warnings block creation |
| `standard` | Errors block, warnings logged |
| `permissive` | Syntax-only validation |

#### 5.4 Expression Compilation

For performance, expressions should be compiled to executable form:

```
Source: click_count / NULLIF(impression_count, 0)
        │
        ▼
    Parse AST
        │
        ▼
    Resolve References
        │
        ▼
    Type Check
        │
        ▼
    Optimize (constant folding, etc.)
        │
        ▼
    Generate SQL/Compute Plan
```

---

## 6. Versioning System

### Requirements

#### 6.1 Immutable Schemas

- Feature schemas are immutable once created
- Schema changes require creating a new version
- Versions follow pattern: `v1`, `v2`, `v3`, ...

#### 6.2 Version Resolution

```python
feature("user_embedding")      # Returns latest version
feature("user_embedding@v1")   # Returns specific version
feature("user_embedding@v2")   # Returns specific version
```

#### 6.3 Version Metadata

```sql
-- Version tracking
SELECT * FROM features
WHERE group_id = ? AND name = 'user_embedding'
ORDER BY version DESC;

-- Latest version
SELECT * FROM features
WHERE group_id = ? AND name = 'user_embedding'
ORDER BY version DESC LIMIT 1;
```

#### 6.4 Version Comparison

The Analytics Service must support version diff analysis:

```python
VersionDiff(
    feature="user_embedding",
    version_a="v1",
    version_b="v2",
    compare=["schema", "distribution"]
)
```

**Comparison Metrics:**
- Schema changes (dtype, nullable, default)
- Row count delta
- Null rate delta
- Distribution metrics (PSI, KL divergence)
- Statistical tests

---

## 7. Lineage Tracking

### Requirements

#### 7.1 Automatic Lineage Extraction

When a derived feature is created, parse the expression and extract all referenced features:

```python
derived_from = "click_count / NULLIF(impression_count, 0)"
# Extracts: [click_count, impression_count]
```

#### 7.2 Lineage Graph Storage

Use a graph database or adjacency list in relational DB:

```sql
-- Adjacency list approach
INSERT INTO feature_lineage (feature_id, upstream_feature_id)
VALUES
    ('ctr_id', 'click_count_id'),
    ('ctr_id', 'impression_count_id');
```

#### 7.3 Lineage Queries

| Query | Description |
|-------|-------------|
| Direct upstream | Features this feature depends on |
| Direct downstream | Features that depend on this feature |
| Transitive upstream | All ancestors |
| Transitive downstream | All descendants |
| Cross-org upstream | External dependencies |

#### 7.4 Lineage Traversal Algorithm

```python
def all_upstream(feature_id: UUID, include_external: bool = False) -> Set[UUID]:
    visited = set()
    queue = [feature_id]

    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        upstreams = get_direct_upstream(current)
        if not include_external:
            upstreams = [u for u in upstreams if same_org(u, feature_id)]

        queue.extend(upstreams)

    return visited - {feature_id}
```

#### 7.5 Lineage Visualization

Support ASCII graph output:

```
click_count ─────────┐
                     ├──▶ ctr ──▶ user_tier
impression_count ────┘
```

---

## 8. Access Control System

### Requirements

#### 8.1 Permission Levels

| Level | Capabilities |
|-------|--------------|
| `reader` | Read feature metadata and data |
| `writer` | Create/update features, write data |
| `admin` | Manage ACLs, delete features, grant external access |

#### 8.2 Cascading Inheritance

ACLs cascade from parent to child by default:

```
Organization ACL
    │ (inherit=true)
    ▼
Domain ACL (merged)
    │ (inherit=true)
    ▼
Project ACL (merged)
    │ (inherit=true)
    ▼
FeatureGroup ACL (merged)
    │ (inherit=true)
    ▼
Feature ACL (merged)
```

#### 8.3 Effective ACL Resolution

```python
def get_effective_acl(resource_id: UUID) -> ACL:
    acl = get_acl(resource_id)

    if not acl.inherit:
        return acl

    parent_acl = get_effective_acl(get_parent_id(resource_id))

    return ACL(
        readers=set(acl.readers) | set(parent_acl.readers),
        writers=set(acl.writers) | set(parent_acl.writers),
        admins=set(acl.admins) | set(parent_acl.admins),
        inherit=True
    )
```

#### 8.4 Permission Check Algorithm

```python
def check_permission(user: str, resource_id: UUID, action: str) -> bool:
    effective_acl = get_effective_acl(resource_id)

    if action == 'read':
        return user in (effective_acl.readers | effective_acl.writers | effective_acl.admins)
    elif action == 'write':
        return user in (effective_acl.writers | effective_acl.admins)
    elif action == 'admin':
        return user in effective_acl.admins

    return False
```

---

## 9. Cross-Organization Access

### Requirements

#### 9.1 Grant Model

External grants allow organizations to share features:

```python
grant_external_access(
    org="partner-org",
    features=["user_embedding", "item_embedding"],  # or ["*"] for all
    permission="read",
    expires_at=datetime(2026, 12, 31)
)
```

#### 9.2 Grant Authorization

- Only resource owners can grant external access
- Grants must be approved by owner (single-party approval)
- Grants can have expiration dates
- Grants can be revoked at any time

#### 9.3 Cross-Org Reference Resolution

```
Reference: @partner-org/ml/rec/items.embedding
           │
           ▼
    Parse: org=partner-org, domain=ml, project=rec, group=items, feature=embedding
           │
           ▼
    Check external_grants table for permission
           │
           ▼
    If granted: Resolve to physical storage location
    If not: Raise CrossOrgAccessError
```

#### 9.4 Federated Query Execution

When a derived feature references cross-org features:

1. Validate all external grants exist and are active
2. Generate federated query plan
3. Execute with appropriate credentials
4. Log cross-org access in audit

---

## 10. Audit Logging

### Requirements

#### 10.1 Logged Actions

| Category | Actions |
|----------|---------|
| **Access** | `READ`, `WRITE`, `QUERY` |
| **Schema** | `CREATE`, `UPDATE_SCHEMA`, `UPDATE_METADATA`, `DELETE`, `DEPRECATE`, `CREATE_VERSION` |
| **ACL** | `UPDATE_ACL`, `GRANT_EXTERNAL`, `REVOKE_EXTERNAL` |
| **Lineage** | `UPDATE_DERIVATION`, `LINEAGE_BREAK` |

#### 10.2 Audit Entry Schema

```json
{
    "id": "uuid",
    "timestamp": "2026-02-21T10:30:00Z",
    "org_id": "uuid",
    "actor": "user@example.com",
    "actor_org": "acme",
    "action": "READ",
    "resource_type": "feature",
    "resource_id": "uuid",
    "resource_path": "acme/ml/rec/user-signals/embedding@v1",
    "metadata": {
        "query_hash": "abc123",
        "rows_accessed": 10000
    },
    "ip_address": "192.168.1.1",
    "user_agent": "raise-sdk/1.0"
}
```

#### 10.3 Audit Query Interface

```python
logs = audit.query(
    resource="user-signals/*",
    actions=["READ", "WRITE"],
    since=datetime.now() - timedelta(days=7),
    exclude_actor_orgs=["acme"],  # Track external access
    limit=1000
)
```

#### 10.4 Audit Retention

- Configurable retention period (default: 365 days)
- Immutable option for compliance
- Export to external storage (S3, GCS)
- Streaming export for large datasets

#### 10.5 Audit Alerts

```python
audit.create_alert(
    name="external-access-alert",
    query=AuditQuery(
        resource="user-signals/*",
        exclude_actor_orgs=["acme"]
    ),
    notify=["security@acme.com"],
    channels=["email", "slack"]
)
```

---

## 11. Analytics Engine

### Requirements

#### 11.1 Analysis Types

| Type | Description | Output |
|------|-------------|--------|
| **Aggregation** | Compute metrics on features | `{metric: value}` |
| **Distribution** | Histogram and percentiles | `{histogram: {...}, percentiles: {...}}` |
| **Correlation** | Feature correlation matrix | `{features: [...], matrix: [[...]]}` |
| **VersionDiff** | Compare feature versions | `{schema_changes: {...}, drift: {...}}` |
| **StatTest** | Statistical hypothesis tests | `{p_value, effect_size, ci}` |
| **RecordLookup** | Sample/filter records | `{records: [...]}` |

#### 11.2 Aggregation Metrics

| Metric | Description |
|--------|-------------|
| `count` | Number of non-null values |
| `sum` | Sum of values |
| `avg` | Mean value |
| `min` | Minimum value |
| `max` | Maximum value |
| `stddev` | Standard deviation |
| `variance` | Variance |
| `null_rate` | Fraction of null values |
| `distinct_count` | Cardinality |
| `percentile_XX` | XX-th percentile |

#### 11.3 Aggregation with Time Windows

```sql
-- Window: "7d", group_by: "user_tier"
SELECT
    user_tier,
    COUNT(*) as count,
    SUM(revenue) as sum,
    AVG(revenue) as avg
FROM feature_table
WHERE timestamp >= NOW() - INTERVAL '7 days'
GROUP BY user_tier;
```

#### 11.4 Rolling Aggregations

```sql
-- window: "1d", rolling: true, periods: 30
SELECT
    date,
    SUM(clicks) OVER (ORDER BY date ROWS 29 PRECEDING) as rolling_sum
FROM daily_aggregates
ORDER BY date DESC
LIMIT 30;
```

#### 11.5 Distribution Analysis

```sql
-- Histogram computation
SELECT
    WIDTH_BUCKET(revenue, 0, 1000, 50) as bucket,
    COUNT(*) as count
FROM feature_table
GROUP BY bucket
ORDER BY bucket;

-- Percentiles
SELECT
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY revenue) as p50,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY revenue) as p75,
    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY revenue) as p90,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY revenue) as p95,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY revenue) as p99
FROM feature_table;
```

#### 11.6 Correlation Matrix

```python
def compute_correlation(features: List[str], method: str) -> np.ndarray:
    data = fetch_feature_data(features)

    if method == "pearson":
        return np.corrcoef(data.T)
    elif method == "spearman":
        return scipy.stats.spearmanr(data)[0]
    elif method == "kendall":
        return data.corr(method='kendall').values
```

#### 11.7 Statistical Tests

| Test | Use Case | Output |
|------|----------|--------|
| `ttest` | Compare means of two groups | t-statistic, p-value, effect size |
| `mann_whitney` | Non-parametric mean comparison | U-statistic, p-value |
| `chi_squared` | Categorical independence | chi2, p-value, contingency table |
| `ks_test` | Distribution comparison | KS statistic, p-value |

#### 11.8 Freshness Control

| Policy | Behavior |
|--------|----------|
| `REAL_TIME` | Always compute fresh results |
| `WITHIN(duration)` | Use cache if fresher than duration |
| `CACHED` | Always use cache if available |

**Cache Implementation:**

```python
def analyze(analysis: Analysis, freshness: Freshness) -> AnalysisResult:
    cache_key = hash(analysis.to_dict())

    if freshness.policy != "real_time":
        cached = cache.get(cache_key)
        if cached and freshness.accepts_age(cached.age):
            return cached

    result = execute_analysis(analysis)
    cache.set(cache_key, result, ttl=3600)

    return result
```

#### 11.9 Async Analysis

For long-running analyses:

```python
job = analyze_async(analysis)
# Returns immediately with job_id

# Later:
job.refresh()  # Poll status
if job.status == "completed":
    result = job.result()
```

**Job Queue Implementation:**
- Submit to distributed job queue (Celery, SQS, etc.)
- Store job metadata in `analysis_jobs` table
- Store results in `analysis_results` table
- Support cancellation

---

## 12. Live Tables (CDC)

### Requirements

#### 12.1 CDC Integration

Live tables subscribe to change data capture streams to auto-refresh:

```
Feature Data Changes
        │
        ▼
    CDC Stream (Debezium, etc.)
        │
        ▼
    CDC Processor
        │
        ▼
    Live Table Refresh Trigger
        │
        ▼
    Re-execute Analysis
        │
        ▼
    Update Materialized Data
```

#### 12.2 Refresh Policies

| Policy | Trigger |
|--------|---------|
| `on_change` | CDC event received |
| `hourly` | Cron: `0 * * * *` |
| `daily` | Cron: `0 0 * * *` |
| `cron(expr)` | Custom cron expression |
| `manual` | Only on explicit refresh call |

#### 12.3 Live Table Storage

Live table data stored in optimized format:

```sql
-- Metadata
INSERT INTO live_tables (group_id, name, analysis_spec, refresh_policy, storage_path)
VALUES (?, 'daily_engagement', '{"type": "aggregation", ...}', '{"type": "on_change"}', 's3://bucket/live/daily_engagement/');

-- Data (Parquet files in S3/GCS)
s3://bucket/live/daily_engagement/
    ├── part-00000.parquet
    ├── part-00001.parquet
    └── _metadata
```

#### 12.4 Refresh Execution

```python
def refresh_live_table(live_table_id: UUID, trigger: str):
    live_table = get_live_table(live_table_id)

    # Record start
    event = create_refresh_event(live_table_id, trigger)

    try:
        # Execute analysis
        result = execute_analysis(live_table.analysis_spec)

        # Write to storage
        write_to_storage(live_table.storage_path, result.data)

        # Update metadata
        update_live_table(live_table_id,
            last_refresh=now(),
            row_count=len(result.data)
        )

        event.status = "completed"
    except Exception as e:
        event.status = "failed"
        event.error_message = str(e)

    save_refresh_event(event)
```

#### 12.5 Querying Live Tables

```python
# Basic query
df = live_table.query()

# With filter
df = live_table.query(filter="user_tier = 'gold'")

# With projection
df = live_table.query(columns=["user_tier", "sum", "avg"])

# With limit
df = live_table.query(limit=100)
```

---

## 13. Dashboards

### Requirements

#### 13.1 Dashboard Definition

```json
{
    "name": "engagement-overview",
    "description": "User engagement metrics",
    "parameters": [
        {
            "name": "date_range",
            "type": "date_range",
            "label": "Date Range"
        },
        {
            "name": "tier",
            "type": "dropdown",
            "label": "User Tier",
            "options": ["all", "bronze", "silver", "gold"],
            "default": "all"
        }
    ],
    "charts": [
        {
            "title": "Daily Clicks",
            "analysis": {"type": "aggregation", "feature": "clicks", ...},
            "chart_type": "line",
            "x": "date",
            "y": "sum"
        }
    ],
    "layout": {
        "type": "grid",
        "columns": 2,
        "rows": [
            {"charts": [0, 1]},
            {"charts": [2, 3]}
        ]
    }
}
```

#### 13.2 Chart Types

| Type | Suitable For |
|------|--------------|
| `line` | Time series |
| `bar` | Categorical comparison |
| `histogram` | Distributions |
| `scatter` | Correlations (2 vars) |
| `heatmap` | Correlation matrices |
| `pie` | Proportions |
| `area` | Stacked time series |
| `table` | Raw data |
| `metric` | Single KPI |
| `gauge` | Progress/threshold |
| `box` | Distribution summary |
| `violin` | Distribution comparison |

#### 13.3 Render Formats

| Format | Use Case |
|--------|----------|
| `json` | Generic spec for custom renderers |
| `html` | Self-contained HTML with embedded JS |
| `yaml` | Human-readable spec |

#### 13.4 Parameter Binding

When rendering, parameters are bound to analysis queries:

```python
def render(dashboard: Dashboard, parameters: Dict) -> str:
    for chart in dashboard.charts:
        analysis = chart.analysis.copy()

        # Bind date_range parameter
        if "date_range" in parameters:
            analysis.window = parameters["date_range"]

        # Bind filter parameters
        if "tier" in parameters and parameters["tier"] != "all":
            analysis.filter = f"user_tier = '{parameters['tier']}'"

        chart.data = execute_analysis(analysis)

    return render_template(dashboard)
```

#### 13.5 Publishing

```python
url = dashboard.publish()
# Returns: https://raise.acme.com/dashboards/engagement-overview
```

- Generate unique URL
- Store rendered snapshot
- Set up periodic refresh
- Configure access control

---

## 14. Alerting System

### Requirements

#### 14.1 Alert Definition

```json
{
    "name": "high-null-rate",
    "analysis": {
        "type": "aggregation",
        "feature": "user_embedding",
        "metrics": ["null_rate"]
    },
    "condition": {
        "type": "greater_than",
        "value": 0.05
    },
    "notify": ["data-quality@acme.com"],
    "channels": ["email", "slack"],
    "check_interval": "1h"
}
```

#### 14.2 Condition Types

| Condition | Parameters | Logic |
|-----------|------------|-------|
| `GREATER_THAN` | `value` | `result > value` |
| `LESS_THAN` | `value` | `result < value` |
| `BETWEEN` | `low`, `high` | `low <= result <= high` |
| `OUTSIDE` | `low`, `high` | `result < low OR result > high` |
| `EQUALS` | `value` | `result == value` |
| `NOT_EQUALS` | `value` | `result != value` |
| `PSI_GREATER_THAN` | `value` | Distribution drift PSI > value |
| `P_VALUE_LESS_THAN` | `value` | Statistical significance p < value |
| `KL_DIVERGENCE_GREATER_THAN` | `value` | KL divergence > value |

#### 14.3 Alert Evaluation

```python
def evaluate_alert(alert: Alert):
    # Execute analysis
    result = execute_analysis(alert.analysis)

    # Extract metric value
    metric_value = extract_metric(result, alert.condition.metric)

    # Evaluate condition
    triggered = alert.condition.evaluate(metric_value)

    if triggered:
        send_notifications(alert.notify, alert.channels, alert, result)
        update_alert_stats(alert.id, triggered=True)

    update_alert_last_check(alert.id)
```

#### 14.4 Notification Channels

| Channel | Implementation |
|---------|----------------|
| `email` | SMTP / SendGrid / SES |
| `slack` | Slack Webhook |
| `pagerduty` | PagerDuty API |
| `webhook` | Custom HTTP POST |
| `sms` | Twilio / SNS |

#### 14.5 Alert Scheduling

Alerts are checked on their defined interval:

```python
# Job scheduler (Celery beat, cron, etc.)
for alert in get_active_alerts():
    if should_check(alert):
        schedule_alert_evaluation(alert.id)
```

---

## 15. ETL and Transformations

### 15.1 Overview

The ETL module enables data transformation pipelines that populate features from various data sources. Transformations are defined as Jobs that combine Sources, Transforms, Targets, and Schedules.

**Key Design Principles:**
- Pipelines are low-level infrastructure constructs, hidden from end users
- Users work with high-level Job abstractions
- Jobs have lineage relationships with features
- Support both SQL and Python transformations
- Incremental processing with checkpoint management

### 15.2 Job Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                            Job                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Sources   │  │  Transform  │  │   Target    │             │
│  │  (1 or more)│  │  (SQL/Py)   │  │  (Features) │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│         │                │                │                      │
│         └────────────────┼────────────────┘                      │
│                          │                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Schedule   │  │ Incremental │  │   Quality   │             │
│  │             │  │   Config    │  │   Checks    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
            ┌─────────────────────────┐
            │     Orchestrator        │
            │  (Airflow, Internal)    │
            └─────────────────────────┘
```

### 15.3 Source Types

| Source Type | Description | Use Case |
|-------------|-------------|----------|
| `ObjectStorage` | S3, GCS, Azure Blob | Raw event data, logs |
| `FileSystem` | Local/network files | Development, batch exports |
| `ColumnarSource` | Data warehouse tables | Aggregated data |
| `FeatureGroupSource` | Existing features | Feature derivation |
| `DatabaseSource` | JDBC/ODBC connections | Operational databases |

#### Source Schema

```sql
CREATE TABLE job_sources (
    id UUID PRIMARY KEY,
    job_id UUID NOT NULL REFERENCES jobs(id),
    source_type VARCHAR(32) NOT NULL,
    name VARCHAR(128),
    config JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);
```

### 15.4 Transform Types

#### SQL Transform

```json
{
    "transform_type": "sql",
    "name": "daily_clicks_transform",
    "sql": "SELECT user_id, COUNT(*) as clicks FROM source WHERE event_time >= '{{checkpoint}}' GROUP BY user_id",
    "source_aliases": {"source": "clickstream"},
    "parameters": {}
}
```

**Template Variables:**
- `{{checkpoint}}` - Current checkpoint value
- `{{execution_date}}` - Logical execution date
- `{{run_id}}` - Unique run identifier
- `{{param_name}}` - Custom parameters

#### Python Transform

```json
{
    "transform_type": "python",
    "name": "segment_users",
    "function_name": "segment_users",
    "module_path": "transforms.segmentation",
    "dependencies": ["pandas", "numpy"]
}
```

**Python Function Signature:**
```python
def transform_function(context: TransformContext, data: Any) -> Any:
    # context provides: job_id, run_id, execution_date, checkpoint_value
    # data is the output from source/SQL stage
    return transformed_data
```

#### Hybrid Transform

Executes SQL first, then applies Python post-processing:

```json
{
    "transform_type": "hybrid",
    "sql_transform": { ... },
    "python_transform": { ... }
}
```

### 15.5 Incremental Processing

#### Processing Modes

| Mode | Description | Checkpoint |
|------|-------------|------------|
| `FULL` | Complete recompute every run | None |
| `INCREMENTAL` | Process only new/changed data | Timestamp/offset |
| `APPEND` | Only append new records | Monotonic ID |
| `UPSERT` | Insert or update by key | Timestamp |

#### Checkpoint Schema

```sql
CREATE TABLE job_checkpoints (
    id UUID PRIMARY KEY,
    job_id UUID NOT NULL REFERENCES jobs(id) UNIQUE,
    checkpoint_type VARCHAR(32) NOT NULL,
    checkpoint_value JSONB,
    checkpoint_column VARCHAR(128),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);
```

#### Checkpoint Flow

```
1. Load checkpoint from store
2. Query source with checkpoint filter: WHERE col >= checkpoint
3. Apply optional lookback: WHERE col >= checkpoint - lookback
4. Execute transform
5. Write to target
6. Update checkpoint to max(col) from processed data
7. Commit checkpoint
```

### 15.6 Target Configuration

```json
{
    "feature_group": "user-signals",
    "features": {
        "daily_clicks": "clicks",
        "daily_revenue": "revenue"
    },
    "write_mode": "upsert",
    "key_columns": ["user_id", "date"]
}
```

**Write Modes:**
- `append` - Insert new rows
- `overwrite` - Replace all data
- `upsert` - Update existing, insert new (requires key_columns)

### 15.7 Schedule Types

| Schedule | Expression | Description |
|----------|------------|-------------|
| `CronSchedule` | `"0 2 * * *"` | Standard cron |
| `IntervalSchedule` | `"30m"`, `"6h"` | Fixed interval |
| `OnChangeSchedule` | CDC-triggered | When source changes |
| `ManualSchedule` | None | Manual trigger only |
| `OnceSchedule` | Timestamp | Single execution |

### 15.8 Quality Checks

Quality checks run after transformation:

| Check Type | Parameters | Severity |
|------------|------------|----------|
| `NullCheck` | column, max_null_rate | ERROR |
| `UniqueCheck` | columns | ERROR |
| `RangeCheck` | column, min, max | ERROR |
| `RowCountCheck` | min_rows, max_rows | WARNING |
| `FreshnessCheck` | column, max_age | ERROR |
| `CustomCheck` | function | Configurable |

#### Quality Report Schema

```sql
CREATE TABLE job_quality_reports (
    id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES job_runs(id),
    passed BOOLEAN NOT NULL,
    checks JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);
```

### 15.9 Job Lifecycle

```
DRAFT → ACTIVE → PAUSED → ACTIVE → DEPRECATED
           ↓         ↑
         FAILED ─────┘
```

| Status | Description |
|--------|-------------|
| `DRAFT` | Job created, not deployed |
| `ACTIVE` | Job deployed and scheduled |
| `PAUSED` | Job deployed but not running |
| `FAILED` | Job failed, requires intervention |
| `DEPRECATED` | Job retired |

### 15.10 Job Metadata Schema

```sql
CREATE TABLE jobs (
    id UUID PRIMARY KEY,
    org_id UUID NOT NULL REFERENCES organizations(id),
    name VARCHAR(128) NOT NULL,
    description TEXT,
    owner VARCHAR(256),
    tags VARCHAR(64)[] DEFAULT '{}',
    sources JSONB NOT NULL DEFAULT '[]',
    transform JSONB NOT NULL,
    target JSONB NOT NULL,
    schedule JSONB NOT NULL,
    incremental JSONB NOT NULL DEFAULT '{}',
    retries INT DEFAULT 3,
    retry_delay INTERVAL DEFAULT '5 minutes',
    timeout INTERVAL DEFAULT '1 hour',
    alerts VARCHAR(256)[] DEFAULT '{}',
    status VARCHAR(32) DEFAULT 'draft',
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE(org_id, name)
);

CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_tags ON jobs USING GIN(tags);
```

### 15.11 Job Run Schema

```sql
CREATE TABLE job_runs (
    id UUID PRIMARY KEY,
    job_id UUID NOT NULL REFERENCES jobs(id),
    status VARCHAR(32) NOT NULL,
    execution_date TIMESTAMP NOT NULL,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    rows_read BIGINT DEFAULT 0,
    rows_written BIGINT DEFAULT 0,
    checkpoint_before JSONB,
    checkpoint_after JSONB,
    metrics JSONB DEFAULT '{}',
    error TEXT
);

CREATE INDEX idx_runs_job ON job_runs(job_id);
CREATE INDEX idx_runs_status ON job_runs(status);
CREATE INDEX idx_runs_execution_date ON job_runs(execution_date);
```

### 15.12 Observability

#### Standard Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `transform_rows_read` | Counter | Rows read from sources |
| `transform_rows_written` | Counter | Rows written to target |
| `transform_duration_seconds` | Histogram | Total execution time |
| `transform_source_read_seconds` | Histogram | Source read time |
| `transform_transform_seconds` | Histogram | Transform execution time |
| `transform_sink_write_seconds` | Histogram | Target write time |
| `transform_checkpoint_lag_seconds` | Gauge | Time behind real-time |
| `transform_quality_checks_passed` | Counter | Passed quality checks |
| `transform_quality_checks_failed` | Counter | Failed quality checks |

---

## 16. Airflow Integration

### 16.1 DAG Generation

Jobs are compiled to Airflow DAGs for execution:

```
Job Definition → DAG Generator → Python DAG File → Airflow Deployment
```

#### Generated DAG Structure

```python
with DAG(dag_id="raise_{job_name}", ...) as dag:
    start = EmptyOperator(task_id='start')
    transform = PythonOperator(task_id='transform', ...)
    quality_checks = PythonOperator(task_id='quality_checks', ...)
    end = EmptyOperator(task_id='end')

    start >> transform >> quality_checks >> end
```

### 16.2 Airflow Configuration

```python
AirflowConfig(
    dag_folder="/opt/airflow/dags",
    default_args={
        "owner": "raise",
        "retries": 3,
        "retry_delay": timedelta(minutes=5),
    },
    catchup=False,
    max_active_runs=1,
    tags=["raise", "feature-store"],
    pool="raise_transforms",
    queue="default",
)
```

### 16.3 Schedule Mapping

| Raise Schedule | Airflow Schedule |
|----------------|------------------|
| `CronSchedule("0 2 * * *")` | `schedule_interval="0 2 * * *"` |
| `IntervalSchedule("1h")` | `schedule_interval=timedelta(hours=1)` |
| `ManualSchedule()` | `schedule_interval=None` |
| `OnChangeSchedule()` | External trigger + sensors |

### 16.4 Deployment API

```
POST /v1/jobs/{job_id}/deploy
  → Generates DAG file
  → Writes to DAG folder
  → Returns orchestrator_id, url

POST /v1/jobs/{job_id}/trigger
  → Calls Airflow API: POST /api/v1/dags/{dag_id}/dagRuns
  → Returns run_id

GET /v1/jobs/{job_id}/status
  → Queries Airflow API: GET /api/v1/dags/{dag_id}
  → Returns deployment status, last_run, next_run
```

### 16.5 Airflow REST API Integration

| Operation | Airflow Endpoint |
|-----------|------------------|
| Trigger DAG | `POST /api/v1/dags/{dag_id}/dagRuns` |
| Get DAG Status | `GET /api/v1/dags/{dag_id}` |
| Get DAG Runs | `GET /api/v1/dags/{dag_id}/dagRuns` |
| Pause DAG | `PATCH /api/v1/dags/{dag_id}` (is_paused=true) |
| Delete DAG | `DELETE /api/v1/dags/{dag_id}` |

### 16.6 Future: Pluggable Orchestrators

The orchestrator interface is designed to be extensible:

```python
class Orchestrator(ABC):
    def deploy(self, job: Job) -> DeploymentResult
    def undeploy(self, job: Job) -> bool
    def trigger(self, job: Job, execution_date: datetime) -> str
    def get_status(self, job: Job) -> JobOrchestratorStatus
    def generate_definition(self, job: Job) -> str
```

**Planned Orchestrators:**
- `AirflowOrchestrator` ✓ (implemented)
- `InternalOrchestrator` ✓ (implemented, for dev/test)
- `DagsterOrchestrator` (future)
- `PrefectOrchestrator` (future)

---

## 17. Storage Requirements

### 17.1 Metadata Store

**Technology:** PostgreSQL 14+

**Requirements:**
- ACID transactions for metadata consistency
- JSONB for flexible schema storage
- Full-text search for feature discovery
- Partitioning for audit log scaling
- Read replicas for query scaling

**Sizing:**
- ~1KB per feature (metadata)
- ~500B per audit entry
- Estimate: 100K features = 100MB metadata + 10GB audit/year

### 15.2 Feature Data Store

**Technology:** Columnar storage (Parquet on S3/GCS, Delta Lake, or Apache Iceberg)

**Requirements:**
- Columnar format for analytical queries
- Schema evolution support
- Efficient compression (especially for embeddings)
- Time-travel for version queries
- CDC integration capability

**Sizing:**
- Varies by data volume
- Embeddings: 4 bytes * dimensions * rows
- Estimate: 1M users * 512-dim embedding = 2GB per feature

### 15.3 Analytics Cache

**Technology:** Redis Cluster

**Requirements:**
- Low-latency access (< 10ms)
- TTL support for freshness control
- Cluster mode for scaling
- Persistence for cache warming

**Sizing:**
- ~10KB per cached result
- Estimate: 10K active analyses * 10KB = 100MB

### 15.4 Lineage Graph Store

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **PostgreSQL (adjacency list)** | Simple, ACID, no new infra | Slow transitive queries |
| **Neo4j** | Native graph, fast traversal | New infrastructure |
| **PostgreSQL + recursive CTE** | Good balance | Moderate complexity |

**Recommendation:** Start with PostgreSQL + recursive CTEs, migrate to Neo4j if lineage graphs become very deep (>100 levels).

### 15.5 Blob Storage

**Technology:** S3 / GCS / Azure Blob

**Use Cases:**
- Live table parquet files
- Audit log exports
- Dashboard snapshots
- Large analysis results

### 15.6 Message Queue

**Technology:** Kafka / SQS / Pub/Sub

**Use Cases:**
- CDC events
- Async job queue
- Alert notifications
- Inter-service communication

---

## 18. API Layer

### 16.1 REST API Endpoints

#### Organizations
```
GET    /v1/organizations
POST   /v1/organizations
GET    /v1/organizations/{org}
PATCH  /v1/organizations/{org}
DELETE /v1/organizations/{org}
```

#### Domains
```
GET    /v1/organizations/{org}/domains
POST   /v1/organizations/{org}/domains
GET    /v1/organizations/{org}/domains/{domain}
PATCH  /v1/organizations/{org}/domains/{domain}
DELETE /v1/organizations/{org}/domains/{domain}
```

#### Projects
```
GET    /v1/domains/{domain}/projects
POST   /v1/domains/{domain}/projects
GET    /v1/domains/{domain}/projects/{project}
PATCH  /v1/domains/{domain}/projects/{project}
DELETE /v1/domains/{domain}/projects/{project}
```

#### Feature Groups
```
GET    /v1/projects/{project}/groups
POST   /v1/projects/{project}/groups
GET    /v1/projects/{project}/groups/{group}
PATCH  /v1/projects/{project}/groups/{group}
DELETE /v1/projects/{project}/groups/{group}
```

#### Features
```
GET    /v1/groups/{group}/features
POST   /v1/groups/{group}/features
POST   /v1/groups/{group}/features/bulk
GET    /v1/groups/{group}/features/{feature}
GET    /v1/groups/{group}/features/{feature}/versions
GET    /v1/groups/{group}/features/{feature}@{version}
PATCH  /v1/groups/{group}/features/{feature}
DELETE /v1/groups/{group}/features/{feature}
POST   /v1/groups/{group}/features/{feature}/deprecate
```

#### Lineage
```
GET    /v1/features/{feature}/lineage
GET    /v1/features/{feature}/lineage/upstream
GET    /v1/features/{feature}/lineage/downstream
GET    /v1/features/{feature}/lineage/graph
```

#### ACL
```
GET    /v1/{resource_type}/{resource}/acl
PUT    /v1/{resource_type}/{resource}/acl
GET    /v1/{resource_type}/{resource}/acl/effective
GET    /v1/{resource_type}/{resource}/acl/chain
```

#### External Access
```
GET    /v1/groups/{group}/external-grants
POST   /v1/groups/{group}/external-grants
DELETE /v1/groups/{group}/external-grants/{org}
```

#### Audit
```
POST   /v1/audit/query
GET    /v1/audit/alerts
POST   /v1/audit/alerts
DELETE /v1/audit/alerts/{name}
POST   /v1/audit/export
```

#### Analytics
```
POST   /v1/analytics/analyze
POST   /v1/analytics/analyze/async
GET    /v1/analytics/jobs/{job_id}
GET    /v1/analytics/jobs/{job_id}/result
DELETE /v1/analytics/jobs/{job_id}
GET    /v1/analytics/results/{result_id}
```

#### Live Tables
```
GET    /v1/groups/{group}/live-tables
POST   /v1/groups/{group}/live-tables
GET    /v1/groups/{group}/live-tables/{name}
DELETE /v1/groups/{group}/live-tables/{name}
POST   /v1/groups/{group}/live-tables/{name}/refresh
GET    /v1/groups/{group}/live-tables/{name}/query
GET    /v1/groups/{group}/live-tables/{name}/history
```

#### Dashboards
```
GET    /v1/dashboards
POST   /v1/dashboards
GET    /v1/dashboards/{name}
PATCH  /v1/dashboards/{name}
DELETE /v1/dashboards/{name}
POST   /v1/dashboards/{name}/render
POST   /v1/dashboards/{name}/publish
```

#### Analytics Alerts
```
GET    /v1/analytics/alerts
POST   /v1/analytics/alerts
GET    /v1/analytics/alerts/{name}
PATCH  /v1/analytics/alerts/{name}
DELETE /v1/analytics/alerts/{name}
POST   /v1/analytics/alerts/{name}/test
```

#### Jobs
```
GET    /v1/jobs
POST   /v1/jobs
GET    /v1/jobs/{name}
PATCH  /v1/jobs/{name}
DELETE /v1/jobs/{name}
POST   /v1/jobs/{name}/deploy
POST   /v1/jobs/{name}/undeploy
POST   /v1/jobs/{name}/trigger
GET    /v1/jobs/{name}/status
GET    /v1/jobs/{name}/runs
GET    /v1/jobs/{name}/runs/{run_id}
POST   /v1/jobs/{name}/checkpoint/reset
GET    /v1/jobs/{name}/lineage
GET    /v1/jobs/{name}/dag
```

### 18.2 Authentication

- OAuth2 / OIDC for user authentication
- API keys for service-to-service
- JWT tokens with org/user claims

### 18.3 Rate Limiting

| Tier | Requests/min | Concurrent Jobs |
|------|--------------|-----------------|
| Free | 100 | 2 |
| Standard | 1000 | 10 |
| Enterprise | 10000 | 100 |

### 18.4 Error Responses

```json
{
    "error": {
        "code": "FEATURE_NOT_FOUND",
        "message": "Feature 'user_embedding' not found in group 'user-signals'",
        "details": {
            "group": "user-signals",
            "feature": "user_embedding"
        }
    }
}
```

---

## 19. Non-Functional Requirements

### 19.1 Performance

| Operation | Target Latency (p99) |
|-----------|---------------------|
| Feature metadata read | < 50ms |
| Feature metadata write | < 100ms |
| Bulk feature creation (100) | < 1s |
| Simple aggregation | < 500ms |
| Complex aggregation | < 5s |
| Correlation matrix (10 features) | < 10s |
| Live table query | < 200ms |
| Dashboard render | < 2s |

### 19.2 Availability

- **Target SLA:** 99.9% (8.76 hours downtime/year)
- **Metadata store:** Multi-AZ deployment
- **Analytics:** Graceful degradation (serve cached results)
- **CDC:** At-least-once delivery

### 19.3 Scalability

| Dimension | Initial | Scale Target |
|-----------|---------|--------------|
| Organizations | 100 | 10,000 |
| Features per org | 10,000 | 1,000,000 |
| Concurrent users | 100 | 10,000 |
| Analytics jobs/day | 1,000 | 1,000,000 |
| Audit events/day | 100,000 | 100,000,000 |

### 19.4 Security

- **Encryption at rest:** AES-256 for all data stores
- **Encryption in transit:** TLS 1.3 for all connections
- **Secrets management:** HashiCorp Vault or AWS Secrets Manager
- **Audit:** All access logged, immutable audit trail
- **Network:** VPC isolation, private endpoints

### 19.5 Compliance

- **Data residency:** Support region-specific deployment
- **Data retention:** Configurable per organization
- **Right to deletion:** Support GDPR/CCPA data removal
- **Audit export:** SOC2/ISO27001 compliant exports

### 19.6 Disaster Recovery

- **RPO:** 1 hour (point-in-time recovery)
- **RTO:** 4 hours (full service restoration)
- **Backup:** Daily snapshots, cross-region replication
- **Runbooks:** Documented recovery procedures

---

## Appendix A: Data Type Specifications

### Primitive Types

| Type | Storage | Size | Range |
|------|---------|------|-------|
| `int64` | BIGINT | 8 bytes | -2^63 to 2^63-1 |
| `float32` | REAL | 4 bytes | IEEE 754 single |
| `float64` | DOUBLE | 8 bytes | IEEE 754 double |
| `bool` | BOOLEAN | 1 byte | true/false |
| `string` | VARCHAR | Variable | UTF-8 |
| `string[N]` | VARCHAR(N) | Variable | UTF-8, max N chars |
| `bytes` | BYTEA | Variable | Binary |
| `timestamp` | TIMESTAMPTZ | 8 bytes | Microsecond precision |

### Embedding Types

| Type | Storage | Size |
|------|---------|------|
| `float16[N]` | ARRAY | 2 * N bytes |
| `float32[N]` | ARRAY | 4 * N bytes |
| `float64[N]` | ARRAY | 8 * N bytes |

### Array Types

| Type | Storage | Notes |
|------|---------|-------|
| `dtype[]` | ARRAY | Variable length |
| `dtype[:N]` | ARRAY | Max length N |

### Struct Types

```python
Struct({
    "field1": Int64(),
    "field2": String(),
    "nested": Struct({"a": Float32()})
})
```

Storage: JSONB or nested columnar

---

## Appendix B: SQL Function Support

### Aggregation Functions

```sql
COUNT(x), COUNT(DISTINCT x)
SUM(x), AVG(x)
MIN(x), MAX(x)
STDDEV(x), STDDEV_POP(x), STDDEV_SAMP(x)
VARIANCE(x), VAR_POP(x), VAR_SAMP(x)
PERCENTILE_CONT(p) WITHIN GROUP (ORDER BY x)
PERCENTILE_DISC(p) WITHIN GROUP (ORDER BY x)
ARRAY_AGG(x), STRING_AGG(x, delimiter)
```

### Math Functions

```sql
ABS(x), SIGN(x)
CEIL(x), FLOOR(x), ROUND(x, n)
TRUNC(x, n)
MOD(x, y), POWER(x, y)
SQRT(x), CBRT(x)
EXP(x), LN(x), LOG(x), LOG10(x)
SIN(x), COS(x), TAN(x)
ASIN(x), ACOS(x), ATAN(x), ATAN2(y, x)
GREATEST(a, b, ...), LEAST(a, b, ...)
```

### Vector Functions (Custom UDFs)

```sql
DOT(vec1, vec2)                    -- Dot product
COSINE_SIMILARITY(vec1, vec2)      -- Cosine similarity
L2_DISTANCE(vec1, vec2)            -- Euclidean distance
L1_DISTANCE(vec1, vec2)            -- Manhattan distance
NORM(vec)                          -- L2 norm
NORMALIZE(vec)                     -- Unit vector
VEC_ADD(vec1, vec2)                -- Element-wise add
VEC_SUB(vec1, vec2)                -- Element-wise subtract
VEC_MUL(vec, scalar)               -- Scalar multiply
```

### String Functions

```sql
CONCAT(a, b, ...), CONCAT_WS(sep, a, b, ...)
LENGTH(s), CHAR_LENGTH(s)
LOWER(s), UPPER(s), INITCAP(s)
TRIM(s), LTRIM(s), RTRIM(s)
LEFT(s, n), RIGHT(s, n)
SUBSTRING(s, start, length)
REPLACE(s, from, to)
SPLIT_PART(s, delimiter, n)
REGEXP_MATCH(s, pattern)
REGEXP_REPLACE(s, pattern, replacement)
```

### Date/Time Functions

```sql
NOW(), CURRENT_TIMESTAMP
DATE_TRUNC(unit, timestamp)
DATE_PART(unit, timestamp)
AGE(timestamp1, timestamp2)
EXTRACT(field FROM timestamp)
timestamp + INTERVAL 'N unit'
timestamp - INTERVAL 'N unit'
TO_CHAR(timestamp, format)
TO_TIMESTAMP(string, format)
```

### Conditional Functions

```sql
CASE WHEN cond THEN result [ELSE default] END
COALESCE(a, b, ...)
NULLIF(a, b)
GREATEST(a, b, ...), LEAST(a, b, ...)
IF(cond, then, else)  -- MySQL compatibility
IIF(cond, then, else) -- SQL Server compatibility
```

### Window Functions

```sql
ROW_NUMBER() OVER (...)
RANK() OVER (...), DENSE_RANK() OVER (...)
LAG(x, n) OVER (...), LEAD(x, n) OVER (...)
FIRST_VALUE(x) OVER (...), LAST_VALUE(x) OVER (...)
NTH_VALUE(x, n) OVER (...)
SUM(x) OVER (...), AVG(x) OVER (...)
-- Window specifications
PARTITION BY col1, col2
ORDER BY col1 [ASC|DESC]
ROWS BETWEEN n PRECEDING AND m FOLLOWING
RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-21 | API Design Team | Initial PRD |

---

*End of Document*
