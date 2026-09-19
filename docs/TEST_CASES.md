# Test Cases and Expected Behaviour

## Router / General
| ID | Input | Expected mode | Expected |
|---|---|---|---|
| ROUTE-001 | Hello, I am Maumita | GENERAL | Friendly conversational answer |
| ROUTE-002 | What is the capital of India? | GENERAL | New Delhi |
| ROUTE-003 | Explain Python decorators | GENERAL | General explanation |
| ROUTE-004 | What is our hotel reimbursement cap? | ENTERPRISE | RAG |
| ROUTE-005 | How many annual leave days do employees receive? | ENTERPRISE | RAG |
| ROUTE-006 | What is our maternity leave policy? | ENTERPRISE | Not-found if absent |
| ROUTE-007 | Who is our CEO? | ENTERPRISE | Not-found if absent |

## Enterprise factual
- Hotel cap → SGD 280/night before taxes → travel_policy.md
- Annual leave → 20 days → employee_handbook.md
- Severity 1 → 30 minutes → product_support.md
- Production data on laptop → prohibited → it_security_policy.md

## Memory
Turn 1: `What is the hotel reimbursement cap?`
Turn 2: `When can it be exceeded?`
Expected: ENTERPRISE route and standalone retrieval query containing hotel/reimbursement.
Turn 3: `Does my manager need to approve that?`
Expected: remains in enterprise travel-policy context.

## Important boundary
`What is the capital of India?` may use general model knowledge.
`What is our maternity leave policy?` must NOT use general model knowledge because it is organisation-specific.
