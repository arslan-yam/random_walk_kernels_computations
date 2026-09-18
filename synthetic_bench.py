"""Compare random-walk kernel estimators on synthetic graph datasets."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import time

from src import gram, utils
from src.benchmark import (KernelConfig,METHODS,add_kernel_arguments,build_inputs,
    compute_kernel,config_from_args,method_seed,option,reference_method,runtime_metadata,
    DIRECT_NODE_LIMIT, CG_REFERENCE_MIN_NODES, planned_methods, method_skip_reason)


def bench(dataset,kind,mu_func,n_graphs,n_samples_mc,n_samples_gvoys,
          distribution_func="uniform",seed=42,labeled=False,*,config=None,methods=None,
          direct_max_nodes=128,sylvester_max_nodes=512,fail_fast=False):
    """Compatibility entry point; all sampling budgets here are TOTAL budgets."""
    graphs = list(dataset)[:n_graphs]
    if not graphs:
        raise ValueError("dataset must not be empty")
    config = config or KernelConfig(kind=kind,lmbd=mu_func(1),n_samples_mc=n_samples_mc,
                                   n_samples_gvoys=n_samples_gvoys)
    config.validate()
    t0 = time.perf_counter()
    Ps,vs,ws = build_inputs(graphs,distribution_func,labeled,seed)
    prep = time.perf_counter()-t0
    matrices,results = {},{}
    max_n = max(map(len,graphs))
    requested = list(methods) if methods is not None else ["direct","cg","fixed_point","series","gvoys","mc"]
    planned = planned_methods(requested, kind=config.kind, max_nodes=max_n)
    for method in planned:
        reason = method_skip_reason(method, kind=config.kind, labeled=labeled,
            max_nodes=max_n, direct_max_nodes=direct_max_nodes,
            sylvester_max_nodes=sylvester_max_nodes)
        if reason:
            results[method] = {"time":None,"err":None,"status":"skipped","reason":reason}
            continue
        print(f"Computing {method} ...",flush=True)
        t0 = time.perf_counter()
        try:
            matrices[method] = compute_kernel(method,Ps,vs,ws,config,method_seed(seed,method),labeled)
            results[method] = {"time":time.perf_counter()-t0,"status":"ok"}
        except (ValueError,RuntimeError,FloatingPointError,MemoryError) as exc:
            results[method] = {"time":time.perf_counter()-t0,"err":None,"status":"failed","error":str(exc)}
            if fail_fast:
                raise
    ref = reference_method(matrices, kind=config.kind, max_nodes=max_n)
    for method,G in matrices.items():
        results[method]["err"] = gram.matrix_errors(matrices[ref],G) if ref else None
    results["metadata"] = {**runtime_metadata(),"kernel":config.metadata(),"seed":seed,
        "distribution":distribution_func,"labeled":bool(labeled),"reference_method":ref,
        "requested_methods":requested,"planned_methods":planned,
        "direct_max_nodes":min(direct_max_nodes,DIRECT_NODE_LIMIT),
        "cg_reference_min_nodes":CG_REFERENCE_MIN_NODES,
        "input_preparation_sec":prep,"n_graphs":len(graphs),
        "node_counts":[len(g) for g in graphs],"edge_counts":[g.number_of_edges() for g in graphs]}
    return results,matrices


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_kernel_arguments(parser,default_mc=100)
    option(parser,"n_graphs",type=int,default=10)
    option(parser,"n_nodes",type=int,default=128)
    option(parser,"graph_type",choices=["er","ba","ws","sbm"],default="er")
    option(parser,"graph_seed",type=int,default=0,help="Separate graph-generation seed; keep fixed to measure estimator noise.")
    option(parser,"p_er",type=float)
    option(parser,"ba_m",type=int,help="Fixed attachment count for sparse BA scaling.")
    option(parser,"ws_k",type=int,help="Fixed ring degree for sparse WS scaling.")
    option(parser,"labeled",type=int,nargs="?",const=1,choices=[0,1],default=0)
    option(parser,"n_labels",type=int,default=3)
    option(parser,"mc_budget_mode",choices=["total","per-node"],default="per-node")
    option(parser,"methods",nargs="+",choices=METHODS,default=["direct","cg","fixed_point","series","gvoys","mc"])
    option(parser,"direct_max_nodes",type=int,default=DIRECT_NODE_LIMIT,
           help="Skip direct at or above this vertex count (0..128); 0 disables direct.")
    option(parser,"sylvester_max_nodes",type=int,default=512)
    option(parser,"output_dir",default="results_v2/synthetic")
    option(parser,"output",help="Exact JSON path; existing files require --overwrite.")
    option(parser,"experiment_name",default="run")
    option(parser,"save_grams",action="store_true")
    option(parser,"overwrite",action="store_true")
    option(parser,"fail_fast",action="store_true")
    args = parser.parse_args(argv)
    if args.n_nodes<2 or args.n_graphs<1 or args.n_labels<1 or args.seed<0 or args.graph_seed<0:
        parser.error("require n_nodes>=2, n_graphs>=1, n_labels>=1 and nonnegative seeds")
    if not 0 <= args.direct_max_nodes <= DIRECT_NODE_LIMIT:
        parser.error("direct_max_nodes must be between 0 and 128")
    return args


def main(argv=None):
    args = parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(args.output) if args.output else Path(args.output_dir)/f"{args.experiment_name}_{args.graph_type}_{args.kind}_seed={args.seed}_{stamp}.json"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; choose another output or --overwrite")
    kwargs = dict(kind=args.graph_type,p_er=args.p_er,ba_m=args.ba_m,ws_k=args.ws_k)
    if args.labeled:
        graphs = [utils.graph_generator_labeled(args.n_nodes,n_labels=args.n_labels,
                  seed=args.graph_seed+i,**kwargs) for i in range(args.n_graphs)]
    else:
        graphs = [utils.graph_generator(args.n_nodes,seed=args.graph_seed+i,**kwargs) for i in range(args.n_graphs)]
    config = config_from_args(args,graphs,args.n_nodes if args.mc_budget_mode=="per-node" else 1)
    print(json.dumps(config.metadata(),indent=2),flush=True)
    results,matrices = bench(graphs,args.kind,utils.mu_func_gen(args.kind,config.lmbd),
        args.n_graphs,config.n_samples_mc,args.n_samples_gvoys,args.u_w_distribution,args.seed,
        bool(args.labeled),config=config,methods=args.methods,direct_max_nodes=args.direct_max_nodes,
        sylvester_max_nodes=args.sylvester_max_nodes,fail_fast=args.fail_fast)
    results["metadata"]["cli"] = vars(args)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(results,indent=2,allow_nan=False)+"\n")
    if args.save_grams:
        with output.with_suffix(".pickle").open("wb") as f:
            pickle.dump(matrices,f)
    print(f"Saved {output}",flush=True)
    return 1 if any(x.get("status")=="failed" for x in results.values() if isinstance(x,dict)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
