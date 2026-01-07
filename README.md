Problem Statement — What traditional systems don’t do (and what we did)
Most e commerce stacks still treat reviews as opaque text blobs or simple aggregates (averages, counts). That approach blocks end to end intelligence because it (1) misses structured extraction of actionable units, (2) cannot trace or audit AI decisions, (3) fails to serve persona specific decision points, and (4) depends on third party inference, inflating latency/cost and risking data governance.
Where legacy breaks down (technically):
•	Unstructured outputs: Keyword lists and star means are not a contract; you can’t reliably pipe them into downstream analytics without brittle regex or ad hoc ETL. There’s no standard payload for pros/cons, verdict, sentiment, key issues, or FAQs, so BI layers struggle to materialize KPIs beyond averages. [Sprint Documentation | PDF]
•	No AI traceability: Typical systems don’t store raw LLM output, prompt version, model/provider, tokens in/out, or latency next to relational facts. That kills reproducibility and regression debugging—you can’t answer “why did the summary change?” or “which prompt generated this verdict?” at audit time. [Sprint Documentation | PDF]
•	Persona agnostic UX: Buyers, sellers, support, and managers need different signals—buyers want concise summaries/FAQs; sellers want pros/cons top N and complaint distributions; support needs severe review triage with sparklines; managers need trend lines, donut distributions, and mismatch heuristics (rating vs. text). Legacy UIs rarely deliver per role metrics & escalation paths. [Sprint Documentation | PDF]
•	Externalized inference: Cloud only LLM calls add per request cost, data egress risk, and latency variability. Without PEFT/LoRA adapters on local GPU, teams can’t control TCO or enforce zero external dependency for sensitive review data. [Sprint Documentation | PDF]
What we engineered (end to end, audit first):
•	A contracted pipeline: Datasets → preprocessing → LoRA fine tuned DeepSeek → JSON outputs → PostgreSQL JSONB + pgvector → FastAPI contracts → React SPA. Each review yields one LLM analysis row with summary, pros, cons, sentiment, verdict, key themes, plus raw_output, prompt_version, llm_model/provider, tokens_input/output, latency_ms for full traceability. (…[PostgreSQL: analyses, product_analyses, product_questions, product_answers, product_embeddings]) [data-1767592899860 | Excel], [Sprint Documentation | PDF]
•	Local GPU inference (PEFT/LoRA): We load DeepSeek 1.3B Instruct and inject adapters for task specific instruction following; outputs respect a fixed JSON schema that downstream services and dashboards consume deterministically. (…[LLM Colab notebook; PEFT adapter load]) [Sprint Documentation | PDF]
•	Role APIs and contracts: Buyer gets /buyer/product/:id/summary; Seller gets /seller/product/:id/insights; Support gets /support/product/:id/urgent; Manager gets /dashboards/product/:id/sentiment-trend and /dashboards/product/:id/key-issues. These FastAPI contracts standardize payloads for each persona and enable non breaking frontend rendering. (…[FastAPI: /buyer/product/:id/summary, /dashboards/, /support/]) [Sprint Documentation | PDF]
•	Persona aware React SPA: Customer.jsx, ProductDetails.jsx, Seller.jsx, Support.jsx, Manager.jsx render audit ready insights with pure SVG charts and defensive fetch/normalization, including FAQ shape reconciliation and image fallbacks. (…[React pages: Customer.jsx, ProductDetails.jsx, Seller.jsx, Support.jsx, Manager.jsx]) [Sprint Documentation | PDF]
Result: We moved from static text and averages to a governed intelligence fabric: structured review analytics, provable AI lineage, persona targeted KPIs, and cost controlled local inference, all backed by PostgreSQL JSONB/pgvector and FastAPI schemas that the React SPA consumes without brittle coupling. (…[Sprint Documentation | PDF])
 
get slide content from copilot, and paste it as bullets on our ppt
 
2) Abstract (expanded)
The Review Intelligence System (RIS) operates as a contract driven analytics pipeline that begins with e commerce datasets—products, reviews, and Q\&A—and terminates in persona specific decision surfaces. Ingested data is normalized for relational integrity and LLM readiness, then passed through a local GPU inference layer powered by PEFT/LoRA adapters on DeepSeek Coder 1.3B Instruct to produce strict, machine readable JSON: summary, pros[], cons[], sentiment, verdict, and key_issues[]. The system enforces schema determinism through instruction prompts and validates outputs before persisting them into PostgreSQL JSONB, co storing traceability fields—raw_output, prompt_version, llm_model, provider, tokens_input/output, latency_ms—to guarantee auditability and reproducibility across releases (…[LLM Colab notebook; PEFT/LoRA adapter upload; PostgreSQL JSONB]).
A service aggregation layer composes materialized views and role APIs: buyers receive product level summaries with pros/cons and avg rating; sellers get insights (sentiment distribution, trend points, complaint tallies); support teams receive severity feeds and sparklines for urgent triage; managers consume KPI cards, line/donut charts, and a mismatch heuristic (rating vs. text) for quality risk tracking (…[FastAPI endpoints: /buyer/product/:id/summary, /seller/product/:id/insights, /dashboards/product/:id/sentiment-trend, /dashboards/product/:id/key-issues, /support/product/:id/urgent]).
The React SPA renders these contracts defensively: pages use inline light theme overrides, pure SVG charting for deterministic rendering, and FAQ normalization to absorb backend shape variations without runtime breakage. Buyer facing pages show succinct summaries and FAQ toggles; Seller surfaces visualize pros/cons top N and trendlines; Support consoles highlight severe review chips, progress bars, and urgent tables; Manager consoles present KPI, donut distributions, and line trends with legends. The application favors non blocking fetch patterns, error fallbacks (image and payload), and role aware navigation that aligns with JWT claims issued by the backend (…[React role pages: Customer.jsx, ProductDetails.jsx, Seller.jsx, Support.jsx, Manager.jsx; FastAPI endpoints]).
Overall, RIS replaces ad hoc text mining with a governed intelligence substrate: curated instruction prompts, LoRA adapted inference under full cost/control, contract validated JSON outputs with telemetry, and persona APIs that separate concerns between content extraction and decision rendering—yielding a composable, enterprise ready analytics stack (…[LLM Colab notebook; FastAPI endpoints; PostgreSQL JSONB; React role pages]).
________________________________________
3) Preparing instruction style datasets (LLM): products, reviews, product answers, product questions (expanded)
Data sources & schema contracts
RIS stages canonical product master and review facts, plus Q\&A pairs for downstream prompt conditioning and RAG style augmentation.
•	Products master: minimal, stable identifiers and attributes—product_id, product_title, brand, category, subtype, price, sku/asin, and a seller link; grain is one product per row (…[products 2000.csv]).
•	Reviews: review_id, FK product_id, rating, review_title, review_text, review_date, reviewer_name, verified_purchase, helpful_votes, language; dates/language normalized for time series and prompt localization (…[reviews 10000.csv]).
•	Product Q\&A: product_id × question_text composite key with guaranteed 1:1 answer integrity; stores answer_text and timestamps for recency preference (…[product questions 8000.csv], [product answers 8000.csv]).
•	Schema dictionary: authoritative listing of domain tables (e.g., analyses, product_analyses, product_questions, product_answers, product_embeddings, reviews, products, sellers, support_teams, users) and staging assets, used to verify types, keys, and relationships (…[data-1767592899860.csv], [Sprint Documentation | PDF]).
Instruction formatting (LLM)
Each review row is transformed into an instruction style prompt consisting of:
1.	Task header (explicit extraction contract),
2.	Input block (verbatim review text, optionally with product metadata for disambiguation), and
3.	Response scaffold (fixed JSON schema to enforce field presence and types).
This scaffold ensures the decoder emits a deterministic structure at generation time—summary, pros[], cons[], sentiment, verdict, key_issues[]—reducing post processing overhead and eliminating fragile regex pipelines. Tokenization uses right padding and length caps to stabilize batch collation; labels mirror input ids for causal LM training (next token prediction), which makes the model learn both format and content fidelity in a single objective (…[LLM Colab notebook cells: prompt builder; tokenizer; prepare labels]).
LLM ready normalization
Before prompt construction, reviews pass through text preprocessing (whitespace, punctuation, Unicode normalization), date canonicalization, language codes, and verified_purchase coercion. For Q\&A, deduplication on (product_id, question_text) and optional recency scoring allow instruction priming (e.g., appending the most recent authoritative answer for context). The final prepared batches—products for grounding, reviews for extraction, and Q\&A for contextual hints—feed the LoRA training and later inference paths with consistent, contractable inputs (…[products 2000.csv], [reviews 10000.csv], [product questions 8000.csv], [product answers 8000.csv], [LLM Colab notebook cells]).
 
2) Abstract (expanded)
The Review Intelligence System (RIS) operates as a contract driven analytics pipeline that begins with e commerce datasets—products, reviews, and Q\&A—and terminates in persona specific decision surfaces. Ingested data is normalized for relational integrity and LLM readiness, then passed through a local GPU inference layer powered by PEFT/LoRA adapters on DeepSeek Coder 1.3B Instruct to produce strict, machine readable JSON: summary, pros[], cons[], sentiment, verdict, and key_issues[]. The system enforces schema determinism through instruction prompts and validates outputs before persisting them into PostgreSQL JSONB, co storing traceability fields—raw_output, prompt_version, llm_model, provider, tokens_input/output, latency_ms—to guarantee auditability and reproducibility across releases (…[LLM Colab notebook; PEFT/LoRA adapter upload; PostgreSQL JSONB]).
A service aggregation layer composes materialized views and role APIs: buyers receive product level summaries with pros/cons and avg rating; sellers get insights (sentiment distribution, trend points, complaint tallies); support teams receive severity feeds and sparklines for urgent triage; managers consume KPI cards, line/donut charts, and a mismatch heuristic (rating vs. text) for quality risk tracking (…[FastAPI endpoints: /buyer/product/:id/summary, /seller/product/:id/insights, /dashboards/product/:id/sentiment-trend, /dashboards/product/:id/key-issues, /support/product/:id/urgent]).
The React SPA renders these contracts defensively: pages use inline light theme overrides, pure SVG charting for deterministic rendering, and FAQ normalization to absorb backend shape variations without runtime breakage. Buyer facing pages show succinct summaries and FAQ toggles; Seller surfaces visualize pros/cons top N and trendlines; Support consoles highlight severe review chips, progress bars, and urgent tables; Manager consoles present KPI, donut distributions, and line trends with legends. The application favors non blocking fetch patterns, error fallbacks (image and payload), and role aware navigation that aligns with JWT claims issued by the backend (…[React role pages: Customer.jsx, ProductDetails.jsx, Seller.jsx, Support.jsx, Manager.jsx; FastAPI endpoints]).
Overall, RIS replaces ad hoc text mining with a governed intelligence substrate: curated instruction prompts, LoRA adapted inference under full cost/control, contract validated JSON outputs with telemetry, and persona APIs that separate concerns between content extraction and decision rendering—yielding a composable, enterprise ready analytics stack (…[LLM Colab notebook; FastAPI endpoints; PostgreSQL JSONB; React role pages]).
________________________________________
3) Preparing instruction style datasets (LLM): products, reviews, product answers, product questions (expanded)
Data sources & schema contracts
RIS stages canonical product master and review facts, plus Q\&A pairs for downstream prompt conditioning and RAG style augmentation.
•	Products master: minimal, stable identifiers and attributes—product_id, product_title, brand, category, subtype, price, sku/asin, and a seller link; grain is one product per row (…[products 2000.csv]).
•	Reviews: review_id, FK product_id, rating, review_title, review_text, review_date, reviewer_name, verified_purchase, helpful_votes, language; dates/language normalized for time series and prompt localization (…[reviews 10000.csv]).
•	Product Q\&A: product_id × question_text composite key with guaranteed 1:1 answer integrity; stores answer_text and timestamps for recency preference (…[product questions 8000.csv], [product answers 8000.csv]).
•	Schema dictionary: authoritative listing of domain tables (e.g., analyses, product_analyses, product_questions, product_answers, product_embeddings, reviews, products, sellers, support_teams, users) and staging assets, used to verify types, keys, and relationships (…[data-1767592899860.csv], [Sprint Documentation | PDF]).
Instruction formatting (LLM)
Each review row is transformed into an instruction style prompt consisting of:
1.	Task header (explicit extraction contract),
2.	Input block (verbatim review text, optionally with product metadata for disambiguation), and
3.	Response scaffold (fixed JSON schema to enforce field presence and types).
This scaffold ensures the decoder emits a deterministic structure at generation time—summary, pros[], cons[], sentiment, verdict, key_issues[]—reducing post processing overhead and eliminating fragile regex pipelines. Tokenization uses right padding and length caps to stabilize batch collation; labels mirror input ids for causal LM training (next token prediction), which makes the model learn both format and content fidelity in a single objective (…[LLM Colab notebook cells: prompt builder; tokenizer; prepare labels]).
LLM ready normalization
Before prompt construction, reviews pass through text preprocessing (whitespace, punctuation, Unicode normalization), date canonicalization, language codes, and verified_purchase coercion. For Q\&A, deduplication on (product_id, question_text) and optional recency scoring allow instruction priming (e.g., appending the most recent authoritative answer for context). The final prepared batches—products for grounding, reviews for extraction, and Q\&A for contextual hints—feed the LoRA training and later inference paths with consistent, contractable inputs (…[products 2000.csv], [reviews 10000.csv], [product questions 8000.csv], [product answers 8000.csv], [LLM Colab notebook cells]).
 
 
 
 
4) Converted into tables (Azure & PostgreSQL) — expanded
Our canonical relational design lands in PostgreSQL for transactional integrity (ACID), native semi structured support (JSONB), and optional vector semantics (pgvector)—with Azure storage (when present) acting as the raw landing zone for CSVs prior to promotion. The promotion pipeline materializes the following core domain tables:
•	products, reviews, analyses, product_analyses, product_questions, product_answers, product_embeddings, support_teams, users, seller_applications (…[data 1767592899860.csv], [Sprint Documentation | PDF]) [data-1767592899860 | Excel], [Sprint Documentation | PDF]
Design specifics (DDL & types):
•	Primary keys on canonical entities (products.id, reviews.id, users.id, etc.), FKs that maintain referential integrity (reviews.product_id → products.id, product_questions.product_id → products.id). (…[data 1767592899860.csv]) [data-1767592899860 | Excel]
•	JSONB columns in analyses (and product_analyses where aggregate views are denormalized) store LLM outputs—summary, pros[], cons[], key_themes[]—plus traceability fields: prompt_version, llm_model, provider, raw_output, tokens_input, tokens_output, latency_ms, created_at for lineage and reproducibility. (…[Sprint Documentation | PDF]) [Sprint Documentation | PDF]
•	product_embeddings: model_name + embedding (JSONB) or vector column (if pgvector is enabled). Indexing strategy may include GIN for JSONB and HNSW for vectors. (…[Sprint Documentation | PDF]) [Sprint Documentation | PDF]
•	Staging (stg_*) tables mirror raw CSV headers for safe ingestion and schema drift isolation; promotion jobs perform type coercion, null handling, and key normalization. (…[data 1767592899860.csv]) [data-1767592899860 | Excel]
Governance & auditability:
•	The use of JSONB alongside strict relational columns lets us store the raw LLM payload and the validated contract simultaneously—supporting versioned prompts and model swaps without breaking the relational interface.
•	Time series fields (created_at, review_date) are standardized at ingestion for downstream materialized views (e.g., daily sentiment rollups). (…[Sprint Documentation | PDF]) [Sprint Documentation | PDF]
Azure → PostgreSQL promotion:
•	Azure blob/file storage holds raw CSVs (products 2000.csv, reviews 10000.csv, product questions 8000.csv, product answers 8000.csv), which are copied into staging tables; a deterministic promotion step validates types against the schema dictionary and upserts into core tables. (…[Sprint Documentation | PDF], [data 1767592899860.csv]) [Sprint Documentation | PDF], [data-1767592899860 | Excel]
________________________________________
5) Data ingestion of customer reviews from the reviews dataset — expanded
Flow: Staging → Normalize → Persist
1.	Staging (stg_reviews): Load raw values from reviews 10000.csv with minimal coercion; preserve external_id, raw rating, review_text, review_date, verified_purchase, helpful_votes, language. (…[reviews 10000.csv]) [Sprint Documentation | PDF]
2.	Normalization:
o	verified_purchase canonicalization (Yes/1/true → true; No/0/false → false).
o	Date/time normalization to ISO (YYYY MM DD or TIMESTAMP WITH TIME ZONE) for time series safety.
o	Optional language codes mapping (e.g., en, hi) for downstream filtering. (…[Sprint Documentation | PDF]) [Sprint Documentation | PDF]
3.	Persist (reviews):
o	Upsert by deterministic keys: (external_id, product_id) ensures idempotency (re runs do not duplicate).
o	FK enforcement: product_id → products.id must exist; reject or quarantine orphan rows. (…[Sprint Documentation | PDF]) [Sprint Documentation | PDF]
Analytical preparation:
•	Build materialized views for sentiment trend (per day: positive/neutral/negative counts) and key issues (top complaint terms/themes aggregated from analyses). These views serve low latency dashboards for support/managers. (…[Sprint Documentation | PDF]) [Sprint Documentation | PDF]
Contracts delivered to frontend:
•	GET /reviews/product/:id: returns flattened reviews (rating, text, date, helpful votes) for UI cards and support tables.
•	GET /dashboards/product/:id/sentiment-trend: exposes daily counts for positive/neutral/negative to drive SVG line charts and legends. (…[/reviews/product/:id], [/dashboards/product/:id/sentiment-trend]) [Sprint Documentation | PDF]
Resilience & observability:
•	Ingestion jobs log row counts, quarantine sets (invalid FK, malformed dates), and idempotent metrics (upserts vs. inserts). This enables replay without side effects and supports audit reporting. (…[Sprint Documentation | PDF]) [Sprint Documentation | PDF]
________________________________________
6) Text preprocessing of reviews — expanded
Normalization operations (pre LLM):
•	Text cleaning: canonicalize quotes (“”→"), punctuation spacing, collapse duplicate whitespace, strip residual HTML tags, and normalize Unicode (NFC/NFKC) to prevent tokenizer surprises.
•	Rating labels: coerce to bounded integer domain (1..5) with guard rails for missing or malformed values; store raw rating separately if necessary.
•	Date unification: accept multiple input formats, convert to ISO date, and ensure TZ awareness when time anchors matter (KPIs, trend charts).
•	Language mapping: map language hints to ISO codes (e.g., en, te, hi) for routing into model prompts and downstream filters. (…[Sprint Documentation | PDF]) [Sprint Documentation | PDF]
Prompt builder coupling:
•	Preprocessed review text is injected into a contracted instruction prompt (task header + input + response scaffold) to force schema determinism on output (summary, pros[], cons[], sentiment, verdict, key_issues[]).
•	Tokenizer settings: right padding, fixed max_length, truncation with guard logs; labels mirror input ids to fit causal LM training/inference expectations. (…[Backend service layer: validation/normalization; LLM prompt construction pipeline]) [Sprint Documentation | PDF]
Fallback heuristics (failure tolerant path):
•	If inference fails or violates schema, run keyword extraction (negative lexicon: “bad”, “poor”, “defect”, etc.) and rating derived heuristics to populate minimal sentiment and key_issues[].
•	Persist the raw model output and fallback marker in JSONB so downstream consumers understand provenance and confidence. (…[Sprint Documentation | PDF]) [Sprint Documentation | PDF]
QA & observability:
•	Maintain preprocessing metrics (language coverage, invalid date ratios, rating coercion counts) to detect data quality drift.
•	Log prompt_version changes and associate them to analysis rows for A/B comparisons during upgrades; use tokens_input/output and latency_ms to profile model throughput and shape future scaling decisions. (…[Sprint Documentation | PDF], [data 1767592899860.csv]) [Sprint Documentation | PDF], [data-1767592899860 | Excel]
Net effect: deterministic tokenization, robust prompt formation, and graceful degradation when LLMs misbehave—keeping the pipeline audit ready and dashboards continuously reliable.
 
 
 
 
7) Fine tuning DeepSeek 1.3B Instruct with LoRA — expanded
Our training stack uses PEFT/LoRA on DeepSeek Coder 1.3B Instruct to achieve adapter level specialization for review intelligence while keeping the base weights frozen. We configure 4 bit NF4 quantization for parameter storage and FP16 compute for kernels, which reduces memory pressure sufficiently for single GPU training/inference while preserving numerical stability in attention blocks (…[LLM Colab notebook cells: BitsAndBytesConfig, prepare_model_for_kbit_training, LoraConfig, TrainingArguments, Trainer, save_pretrained, upload_folder]; [Sprint Documentation | PDF]). [Sprint Documentation | PDF]
Adapter targets & hyperparameters. We attach LoRA adapters to the attention projections (q_proj, k_proj, v_proj, o_proj), since most representational power for instruction following and structured JSON generation lives in self attention. We set rank r=8, α=16 (conventional ≈ 2×r), and dropout=0.05 to limit overfitting on templated responses. Training uses the Causal LM objective, so labels = input_ids, ensuring the model learns the output format contract (JSON keys/structure) alongside semantic extraction. We cap sequence length at 512 with right padding and truncation guards; this balances coverage of long reviews with throughput. Gradient accumulation creates an effective batch size larger than per device memory would allow, and ~3 epochs strikes a good trade off between adaptation and generalization given our corpus (…[LLM Colab notebook; Sprint Documentation | PDF]). [Sprint Documentation | PDF]
Optimization & instrumentation. The training loop is orchestrated via HF Trainer, inheriting mixed precision, logging intervals, and epoch wise checkpointing. We track loss curves, tokens_input/output, and latency per step to diagnose bottlenecks and confirm convergence. On completion, we persist adapter_model.safetensors and adapter_config.json (adapter graph and hyperparameters). We then upload these artifacts to Hugging Face for versioning and roll back safety, enabling controlled promotion through environments (…[LLM Colab notebook cells: save_pretrained, upload_folder]; [Sprint Documentation | PDF]). [Sprint Documentation | PDF]
Inference readiness (runtime path). In production, we load the base DeepSeek model, inject the LoRA adapter via PEFT, and place the composite onto GPU. Prompts follow our instruction → input → response scaffold pattern to force deterministic JSON (summary, pros[], cons[], verdict, sentiment, key_issues[]). We capture traceability fields (model id/provider/prompt version/tokens/latency) alongside the validated payload for audit, regression, and cost profiling (…[LLM Colab inference cell; local GPU runtime]; [Sprint Documentation | PDF]). [Sprint Documentation | PDF]
Why this design is resilient.
•	Cost/control: No per call external API; adapters upgrade independently of base weights.
•	Governance: Persisted raw_output and prompt_version let us reproduce any inference and explain changes.
•	Performance: NF4 + FP16 and single GPU keep latency within SLA, suitable for batch and near real time flows (…[Sprint Documentation | PDF]). [Sprint Documentation | PDF]
________________________________________
8) Inference (end to end data flow) — expanded
1) Prompt construction. For each review (optionally enriched with minimal product context: title/brand/category), the backend prompt builder composes the instruction block, the input block (verbatim review text), and a response scaffold that mirrors our fixed JSON schema. This standardization prevents schema drift and minimizes post generation parsing complexity (…[Sprint Documentation | PDF]). [Sprint Documentation | PDF]
2) Local GPU inference (DeepSeek + LoRA). The runtime hosts the base DeepSeek weights and injects LoRA adapters via PEFT. Generation runs on GPU using our tokenization configuration (right padding, 512 tokens), and the decoder yields machine readable JSON with fields: summary, pros[], cons[], key_issues[], verdict, sentiment. We record latency and token counts as part of telemetry (…[LLM Colab inference cell; local GPU runtime]; [Sprint Documentation | PDF]). [Sprint Documentation | PDF]
3) Validation & persistence. The backend validates the JSON against the contract (keys present, types correct; optional normalization of enums like sentiment). We store the validated payload plus traceability into PostgreSQL: analyses row per review with JSONB for the structured fields, and columns for raw_output, llm_model, provider, prompt_version, tokens_input, tokens_output, latency_ms, and created_at. This preserves full lineage for audits and supports A/B comparisons across prompt or adapter versions (…[data 1767592899860 | Excel]). [data-1767592899860 | Excel]
4) Aggregations & persona contracts.
•	Buyer summaries: /buyer/product/:id/summary aggregates pros/cons, computes avg rating, and emits display ready summary/verdict.
•	Seller insights: /seller/product/:id/insights composes sentiment distribution, trend points (daily positive/neutral/negative), and top complaints from key_issues[].
•	Dashboards: /dashboards/product/:id/sentiment-trend exposes daily counts; /dashboards/product/:id/key-issues returns ranked themes.
•	Support feeds: /support/product/:id/urgent prioritizes negative or high severity reviews with sparklines and helpful vote context (…[Sprint Documentation | PDF]). [Sprint Documentation | PDF]
Frontend consumption.
•	ProductDetails.jsx pulls buyer summary/FAQs; SellerProductDetails.jsx renders pie/line/bar (SVG); Support.jsx/SupportProduct.jsx display severity chips + sparklines; Manager.jsx shows KPI + line/donut charts. All views call the contracts above and render defensively (FAQ normalization, image fallbacks) (…[Sprint Documentation | PDF]). [Sprint Documentation | PDF]
Outcomes.
•	Determinism in output structure → minimal parsing failures.
•	Traceability at row level → reproducible investigations.
•	Persona ready APIs → low coupling between extraction and UX.
•	Operational resilience via local inference and stored telemetry (…[data 1767592899860 | Excel], [Sprint Documentation | PDF]).
9) Customer View — Buyer Perspective (Product page consumption) (expanded)
The buyer flow is optimized for low latency decisioning on the product detail page. When a user navigates to a product, the frontend issues parallel data fetches: product core (…[/products/by-id/:id]) and buyer summary (…[/buyer/product/:id/summary]); optionally FAQs (…[/faq/product/:id]) are pulled and then normalized client side to absorb backend shape variance. The summary contract exposes a stable payload—summary, pros[], cons[], verdict, and average_rating—rendered in ProductDetails.jsx with pure SVG star widgets and light theme cards. FAQs are de duplicated and canonicalized to handle differences such as { items, data, results, faq }, ensuring button toggle disclosure renders consistently (…[ProductDetails.jsx; /buyer/product/:id/summary; /faq/product/:id]). [Sprint Documentation | PDF]
Technically, the buyer page supports non blocking UI: each panel (summary, verdict, pros/cons) shows loading states independently; image fallbacks avoid broken media (default hero image on onError). The hash anchor #reviews triggers a smooth scroll after the reviews list loads, providing targeted navigation to real customer comments. Review write back uses a guarded POST /reviews/ payload (rating + text, ISO date) with optimistic prepend; errors are surfaced inline and the AuthContext ensures only appropriate roles can post (…[ProductDetails.jsx; /reviews/product/:id; POST /reviews]). [Sprint Documentation | PDF]
The buyer experience is “contract first”: the page trusts the backend JSON contract to be stable, and defensive renderers (FAQ normalization, star value coercion, currency formatting) prevent breakage if optional fields are missing. Materialized metrics (average rating) come from buyer summary aggregation built over analyses rows and raw reviews, ensuring flicker free KPI display while detailed customer content (reviews) streams in. All of this is governed by the audit ready pipeline where traceability lives in JSONB (raw LLM outputs and metadata), so any change in summarization or verdict can be reproduced and investigated (…[Sprint Documentation | PDF]). [Sprint Documentation | PDF]
________________________________________
10) Customer View — Seller / Merchant Perspective (Insights & trendlines) (expanded)
The seller surfaces are split across catalog navigation and product deep dive analytics. Sellers first open SellerProducts.jsx, which presents their own product list from /seller/me/products with text search and an Amazon style brand facet (searchable, counts derived from the text filtered list). Clicking any card navigates to /seller/product/:id, where SellerProductDetails.jsx loads core product meta (…[/products/by-id/:id]) and seller insights (…[/seller/product/:id/insights]) in parallel (…[SellerProducts.jsx; SellerProductDetails.jsx; /seller/me/products; /products/by-id/:id; /seller/product/:id/insights]). [Sprint Documentation | PDF]
Insights payloads include:
•	Sentiment distribution → pie chart (pure SVG) mapping normalized labels (pos/neg/neu) to Positive/Negative/Neutral;
•	Trend points → line chart of date wise counts for positive/neutral/negative;
•	Top complaints → bar chart showing key_issues[] ranked by counts;
•	Pros/cons top N → badge chips delineating product strengths/weaknesses.
These visualizations feed from analyses JSONB, aggregated by product and day, with normalization applied client side for label consistency. The Seller.jsx home also allows choosing a support team via /support/seller/my-support-team and /support/teams, storing the association to route future severe issues to the right triage group (…[Seller.jsx; /support/seller/my-support-team; /support/teams]). [Sprint Documentation | PDF]
From a technical standpoint, all charts are pure SVG (no external charting libs): predictable rendering, small payload, and enterprise compliance. Data readiness is ensured through FastAPI contracts that expose deterministic shapes; the frontend applies defensive normalization so minor backend changes don’t impact rendering. Since each review maps to one analysis row with traceability (prompt, model, tokens), sellers can trust that trends and distributions are reproducible across versions. This view is read only by design; write paths (e.g., listing edits) are considered out of scope here and remain in seller management tools (…[Sprint Documentation | PDF]). [Sprint Documentation | PDF]
________________________________________
11) Customer View — Customer Support Perspective (Escalation) (expanded)
Support teams require priority handling for negative or severe reviews. Support.jsx starts with a team scoped sellers list from /support/me/sellers. Selecting a seller triggers /support/seller/:sellerId/products, which returns product rows annotated with severe_total and a short sparkline series (…[Support.jsx; /support/me/sellers; /support/seller/:sellerId/products]). [Sprint Documentation | PDF]
Each product card renders:
•	A severity chip (tone: green/amber/red) computed from severe_total;
•	A SeverityBar (relative to the max severe count in the filtered grid);
•	A Sparkline (SVG mini trend of severe counts over time).
Clicking Severe navigates to SupportProduct.jsx at /support/products/:id, which fetches product meta (…[/products/by-id/:id]) and the urgent feed (…[/support/product/:id/urgent])—a table of high priority reviews with columns for date, rating, sentiment, severity, top_issue, excerpt, and helpful_votes. Sentiment and severity cells are rendered as chips with dot color per polarity; excerpts are ellipsized for readability. The table has a sticky header, lazy scrolling, and light theme contrasts (…[SupportProduct.jsx; /products/by-id/:id; /support/product/:id/urgent]). [Sprint Documentation | PDF]
This console is tuned for triage speed and audit readiness: the severe feed is derived from analyses (and sometimes review heuristics) with traceability fields intact at the database layer; telemetry helps explain escalation spikes (e.g., prompt version rollouts). Frontend components are intentionally defensive—image fallbacks, null safe dots/chips, and accessible role attributes (list, table, region)—to support mixed datasets. Support can move back to the seller dashboard or to buyer detail pages to cross verify context. All flows are RBAC guarded via AuthContext (…[Sprint Documentation | PDF]). [Sprint Documentation | PDF]
________________________________________
13) Customer View — Business & Product Management Perspective (Dashboards) (expanded)
Managers consume aggregate signals across products. Manager.jsx begins with /products?limit=100 to seed the dropdown, then issues coordinated calls on selection:
•	Buyer summary (…[/buyer/product/:id/summary]) to get average_rating;
•	Urgent feed (…[/support/product/:id/urgent]) to count priority items;
•	Sentiment trend (…[/dashboards/product/:id/sentiment-trend]) for daily series;
•	Key issues (…[/dashboards/product/:id/key-issues]) for ranked complaint themes;
•	Reviews (…[/reviews/product/:id]) to compute platforms and a mismatch heuristic (rating vs negative text cues) (…[Manager.jsx]). [Sprint Documentation | PDF]
The dashboard renders:
•	KPI cards: average rating, urgent count (filtered by minStars), total reviews, mismatch count.
•	LineTrendChart (SVG) with positive/neutral/negative series.
•	DonutChart (SVG) for distribution of sentiments.
•	Lists for key issues and mismatch samples (date + text excerpt).
The mismatch heuristic spotlights potential data quality or review gaming (e.g., 4★ but text contains negative tokens); managers use this as triage for moderation or seller outreach. Backend materialized views ensure trend endpoints remain fast; frontend defensive code avoids failure when optional arrays are empty (show muted dashes instead). The manager console preserves independence from a specific chart library and relies on stable contracts from FastAPI. All underlying analytics ultimately trace back to analyses JSONB and reviews, so leadership can audit any KPI back to individual review rows (…[Sprint Documentation | PDF]). [Sprint Documentation | PDF]
________________________________________
11) Review summary generation, pros, cons, verdict, sentiment classification (expanded)
Contracted JSON fields (per review):
•	summary → concise synthesis of review content; sentence level paraphrase optimized for buyer readability.
•	pros[] / cons[] → salient aspects extracted as short noun phrases, later count aggregated for top N lists in seller views.
•	sentiment → one of positive/neutral/negative; drives daily trend lines and donut distribution.
•	verdict → product level guidance summarizing the likely suitability (e.g., “good for travelers”).
•	key_issues[] → frequent complaint themes; fuels key issue dashboards and support triage.
•	Persisted in JSONB with traceability fields (raw_output, llm_model, provider, prompt_version, tokens_input/output, latency_ms) for audit and regression (…[analyses JSONB; buyer summary aggregation; seller insights aggregation]). [Sprint Documentation | PDF]
Quality & fallback:
When the LLM output violates the contract (missing keys, wrong types), we backstop with keyword extraction (negative lexicon) and rating heuristics to assign a minimal sentiment and populate key_issues[]. The system flags these rows so downstream views can reflect confidence (e.g., muted styling), while still maintaining availability—no blank sections on buyer/seller/support/manager pages (…[AI fallback strategy]). [Sprint Documentation | PDF]
Why this works:
The standardized review JSON allows deterministic aggregation for materialized views and persona APIs, enabling low coupling between extraction and visualization. And because each analysis row carries prompt/version/model metadata, upgrades (new prompts or adapters) can be A/B tested safely, with the ability to reconstruct past decisions from stored raw_output (…[Sprint Documentation | PDF]). [Sprint Documentation | PDF]
________________________________________
References
•	(Sprint Documentation | PDF) architecture & API contracts: buyer/seller/support/manager views, analyses JSONB, materialized views, RBAC. [Sprint Documentation | PDF]
•	(data 1767592899860.csv | schema dictionary) table/column listing for analyses, product_analyses, product_questions/answers, product_embeddings, products, reviews, users, support_teams. [data-1767592899860 | Excel]
 
 
 
