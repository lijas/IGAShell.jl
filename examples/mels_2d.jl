using Five
using IgAShell

#Dimension
const DIM = 3
const NELX = 110
const NELY = 1

const ORDERS = (2,2,2)

const L = 110.0
const h = 8.0
const b = 20.0


data = ProblemData(
    dim = DIM,
    tend = 1.0,
    adaptive = true
)

#Geometric data
angles = T[0,0,0,0]

#Definition for 3d

#IM7/8552 Analysis of Ninety Degree Flexure Tests for Characterization of CompositeTransverse Tensile Strength
E1 = 61.65e3
E2 = 61.65e3
E3 = 13.61e3
G_12 = 23.37e3
G_13 = 4.55e3
G_23 = 4.55e3
ν_12 = 0.3187
ν_13 = 0.3161
ν_23 = 0.3161
rho = 7.8e-6;

#Interface material 
G1 = 211.0/1000; G2 = 1050/1000  
τ1 = 60.0e0;  τ2 = 90.0e0;     


Ne0 = 3
bc_penalty = 1e7

@show τ_approx = sqrt( (9pi*E1*G1)/(32*Ne0*le) )

#interfacematerial = IGAShell.MatCohesive{dim}(λ_0,λ_f,τ,K)
interfacematerial = MatCZBilinear(
    K = 1.0e5,
    τᴹᵃˣ = ( 1050/1000, 1050/1000, 211.0/1000 ), 
    Gᴵ = 0.5.*(90.0, 90.0, 60.0), 
    η = 1.6
) 

material(α) = MatTransvLinearElastic{dim}(
    E1 = 61.65e3, 
    E2 = 61.65e3, 
    E3 = 13.61e3,
    ν_12 = 0.3187, 
    ν_13 = 0.3161, 
    ν_23 = 0.3161, 
    G_13 = 4.55e3, 
    G_12 = 23.37e3, 
    G_23 = 4.55e3,
    plane_stress = false,
    α = α
)  
layermats = [material(α) for α in 1:length(angles)]

#
data.grid = IGAShell.IGA.generate_nurbsmesh((NELX, ), (ORDERS[1], ), (L, ), sdim=dim) |> IGAShell.IGA.convert_to_grid_representation
 
#Sets
addcellset!(grid, "precrackedu", (x) -> x[1] > L-au)
addcellset!(grid, "precrackedl", (x) -> x[1] > L-al)
precracked_l = collect(getcellset(grid, "precrackedl"))
precracked_u = collect(getcellset(grid, "precrackedu"))
precracked_cells = unique(union(precracked_l, precracked_u))

partset1 = collect(1:length(grid.cells))

addvertexset!(grid, "right", (x)-> x[1] ≈ L)
addvertexset!(grid, "left", (x)-> x[1] ≈ 0.0)
addvertexset!(grid, "zfixed", (x)-> x[1] ≈ 9.5)
@assert(length(getvertexset(grid, "zfixed")) > 0)
#@show getindex.(getproperty.(grid.nodes, :x), 1)
#@show length(getvertexset(grid, "zfixed"))

cellstates = [IGAShell.LAYERED for i in 1:prod(nels)]
cellstates[precracked_u] .= IGAShell.STRONG_DISCONTINIUOS_AT_INTERFACE(3)
cellstates[precracked_l] .= IGAShell.STRONG_DISCONTINIUOS_AT_INTERFACES((1,3))

ninterfaces = nlayers - 1

interface_damage = zeros(T, ninterfaces, getncells(grid))
interface_damage[1, precracked_l] .= 1.0
interface_damage[3, precracked_u] .= 1.0

#IGAshell data
igashelldata = 
IGAShell.IGAShellData(;
    layer_materials           = layer_mats,
    interface_material        = interfacematerial,
    viscocity_parameter       = 0.0,
    orders                    = (orders..., r),
    knot_vectors              = nurbsmesh.knot_vectors,
    thickness                 = h,
    width                     = dim == 2 ? b : 1.0,
    initial_cellstates        = cellstates,
    initial_interface_damages = interface_damage,
    nqp_inplane_order         = 3,
    nqp_ooplane_per_layer     = 2,
    adaptable                 = true,
    small_deformations_theory = false,
    LIMIT_UPGRADE_INTERFACE   = 0.03,
    nqp_interface_order       = 4
)  

igashell = 
IGAShell.IGAShell(
    cellset = partset1, 
    connectivity = reverse(nurbsmesh.IEN, dims=1), 
    data = igashelldata
) 
push!(data.parts, igashell)

#
data.output[] = Output(
    interval = 5.0,
    runname = "mels_2d",
    savepath = "./"
)

#
etf = IGAShellWeakBC( 
    set = getvertexset(grid, "left"), 
    func = (x,t) -> zeros(T,dim), 
    comps = 1:DIM,
    igashell = igashell, 
    penalty = bc_penalty
)
push!(data.constraints, etf)

#Supprted bottom
zfixedcell = collect(getvertexset(grid, "zfixed"))[2]
z_fixed_edgeset = VertexInterfaceIndex(zfixedcell..., 1)
etf = IGAShell.IGAShellWeakBC( [z_fixed_edgeset], (x,t) -> [0.0], [dim], igashell, penalty = bc_penalty)
push!(data.constraints, etf)

#Force
edgeset = VertexInterfaceIndex(getvertexset(grid, "right"), 2)
#etf = IGAShell.IGAShellWeakBC(edgeset, (x,t) -> [-t*umax], [dim], igashell, penalty = bc_penalty)
#push!(data.constraints, etf)
etf = IGAShell.IGAShellExternalForce(edgeset, (x,t) -> [zeros(T,dim-1)..., -1.0/b], igashell)
push!(data.external_forces, etf)

data.outputs["forcedofs2"] = IGAShell.IGAShellBCOutput(Ref(igashell), outputset = edgeset, components = [dim], interval = 0.00)

#Stress output
#=postcells = [50, 75, 90]
stress_output = IGAShell.IGAShellStressOutput(Ref(igashell), cellset = postcells, interval = 0.00)
data.outputs["Stress at 50%"] = stress_output
stress_output = IGAShell.IGAShellRecovoredStressOutput(Ref(igashell), cellset = postcells, interval = 0.00)
data.outputs["RS at 50%"] = stress_output=#

solverinput = IGAShell._build_problem(data) do dh, parts, dbc
    instructions = IGAShell.initial_upgrade_of_dofhandler(dh, igashell)
    IGAShell.update_dofhandler!(dh, IGAShell.StateVariables(T, ndofs(dh)), IGAShell.StateVariables(T, ndofs(dh)), IGAShell.SystemArrays(T, ndofs(dh)), instructions)
    
    alldofs = collect(1:ndofs(dh))
    JuAFEM.copy!!(dbc.free_dofs, alldofs)

    #=
    local_locked_dofs = Int[]
    index  = first(getvertexset(grid, "left"))
    cellid, faceid = index
    append!(local_locked_dofs, IGAShell.igashelldofs(igashell, index))

    globaldofs = celldofs(dh, cellid)
    locked_dofs = globaldofs[local_locked_dofs]
    
    #append!(loacked_dofs, IGAShell.igashelldofs(igashell, z_fixed_edgeset))
    alldofs = collect(1:ndofs(dh))

    JuAFEM.copy!!(dbc.prescribed_dofs, locked_dofs)
    JuAFEM.copy!!(dbc.free_dofs, setdiff(alldofs, locked_dofs))
    JuAFEM.copy!!(dbc.values, zeros(T, length(locked_dofs)))
    =#
end

solver = DissipationSolver(Δλ0 = 1e-5, Δλ_max = 10.0, Δλ_min = 1e-8, ΔL0 = 2.5e-0, ΔL_min = 1e-6, ΔL_max = 5e0, sw2d = 1.0, sw2i = 1e-7, optitr = 5, maxitr = 17, maxsteps = 1000, λ_max = 700.0, λ_min = -50.0, tol=1e-4, max_residual=1e5)
#solver = StaticSolver2{dim,T}(0.0, data.tend, 0.01);
#par = ArcLengthSolverParameters(Δλ = 5.0, Δλ_max = 10.0, Δλ_min = 1e-8, ΔL = 5e-0, ΔL_min = 1e-6, ΔL_max = 5e0, sw2d = 1e-1, sw2i = 1e-7, maxitr = 600, λmax = 1000, λmin = -50)
#solver =  ArcLengthSolver{dim,T}(0.0, data.tend, 0.01, par);
return solver, solverinput




output = solvethis(solver, solverinput...)


