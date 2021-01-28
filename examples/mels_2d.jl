using Five
using IgAShell

#Dimension
const DIM = 2
const NELX = 110
const NELY = 1

const ORDERS = (2,2)

const L = 110.0
const h = 8.0
const b = 20.0
const au = 56.0
const al = 32.0

angles = deg2rad.([0.0, 0.0, 0.0, 0.0])
nlayers = length(angles)
ninterfaces = nlayers - 1

data = ProblemData(
    dim = DIM,
    tend = 1.0,
    adaptive = true
)

#interfacematerial = IGAShell.MatCohesive{dim}(λ_0,λ_f,τ,K)
interfacematerial = MatCZBilinear(
    K    = 1.0e5,
    Gᴵ   = (1050/1000, 1050/1000, 211.0/1000 ),
    τᴹᵃˣ = 0.5.*(90.0, 90.0, 60.0),
    η    = 1.6
) 

material(_α) = 
MatTransvLinearElastic(
    E1 = 61.65e3, 
    E2 = 61.65e3, 
    E3 = 13.61e3,
    ν_12 = 0.3187, 
    ν_13 = 0.3161, 
    ν_23 = 0.3161, 
    G_13 = 4.55e3, 
    G_12 = 23.37e3, 
    G_23 = 4.55e3,
    α = _α
) 
layermats = [Material2D(material(α), Five.PLANE_STRAIN) for α in angles]

#
nurbsmesh = IgAShell.IGA.generate_nurbsmesh((NELX, ), (ORDERS[1], ), (L, ), sdim=DIM) 
data.grid = IgAShell.IGA.convert_to_grid_representation(nurbsmesh)

#Sets
addcellset!(data.grid, "precrackedu", (x) -> x[1] > L-au)
addcellset!(data.grid, "precrackedl", (x) -> x[1] > L-al)
precracked_l = collect(getcellset(data.grid, "precrackedl"))
precracked_u = collect(getcellset(data.grid, "precrackedu"))
precracked_cells = unique(union(precracked_l, precracked_u))

partset1 = collect(1:length(data.grid.cells))

addvertexset!(data.grid, "right", (x)-> x[1] ≈ L)
addvertexset!(data.grid, "left", (x)-> x[1] ≈ 0.0)
addvertexset!(data.grid, "zfixed", (x)-> x[1] ≈ 9.5)
@assert(length(getvertexset(data.grid, "zfixed")) > 0)
#@show getindex.(getproperty.(grid.nodes, :x), 1)
#@show length(getvertexset(grid, "zfixed"))

cellstates = [IgAShell.LAYERED for i in 1:NELX]
cellstates[precracked_u] .= IgAShell.STRONG_DISCONTINIUOS_AT_INTERFACE(3)
cellstates[precracked_l] .= IgAShell.STRONG_DISCONTINIUOS_AT_INTERFACES((1,3))
cellstates = [IgAShell.STRONG_DISCONTINIUOS_AT_INTERFACES((1,3)) for i in 1:NELX]

interface_damage = zeros(ninterfaces, NELX)
interface_damage[1, precracked_l] .= 1.0
interface_damage[3, precracked_u] .= 1.0

#IGAshell data
igashelldata = 
IgAShell.IGAShellData(;
    layer_materials           = layermats,
    interface_material        = interfacematerial,
    viscocity_parameter       = 0.0,
    orders                    = ORDERS,
    knot_vectors              = nurbsmesh.knot_vectors,
    thickness                 = h,
    width                     = DIM == 2 ? b : 1.0,
    initial_cellstates        = cellstates,
    initial_interface_damages = interface_damage,
    nqp_inplane_order         = 3,
    nqp_ooplane_per_layer     = 2,
    adaptable                 = false,
    small_deformations_theory = false,
    LIMIT_UPGRADE_INTERFACE   = 0.03,
    nqp_interface_order       = 4
)  

igashell = 
IgAShell.IGAShell(
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
    set = getvertexset(data.grid, "left"), 
    func = (x,t) -> zeros(DIM), 
    comps = 1:DIM,
    igashell = igashell, 
    penalty = 1e7
)
push!(data.constraints, etf)

#Supprted bottom
zfixedcell = collect(getvertexset(data.grid, "zfixed"))[2]
z_fixed_edgeset = VertexInterfaceIndex(zfixedcell..., 1)

etf = IGAShellWeakBC( 
    set = [z_fixed_edgeset], 
    func = (x,t) -> [0.0], 
    comps = [DIM],
    igashell = igashell, 
    penalty = 1e7
)
push!(data.constraints, etf)

#Force
edgeset = VertexInterfaceIndex(getvertexset(data.grid, "right"), 2)
etf = IGAShellExternalForce(
    set = edgeset, 
    func = (x,t) -> [0.0, -1.0/b],
    igashell = igashell
)
push!(data.external_forces, etf)

#data.outputs["forcedofs2"] = IGAShell.IGAShellBCOutput(Ref(igashell), outputset = edgeset, components = [dim], interval = 0.00)


#Stress output
#=postcells = [50, 75, 90]
stress_output = IGAShell.IGAShellStressOutput(Ref(igashell), cellset = postcells, interval = 0.00)
data.outputs["Stress at 50%"] = stress_output
stress_output = IGAShell.IGAShellRecovoredStressOutput(Ref(igashell), cellset = postcells, interval = 0.00)
data.outputs["RS at 50%"] = stress_output=#

#=solver = NewtonSolver(
    Δt0 = 0.1,
    Δt_max = 0.1,

)=#

state, globaldata = build_problem(data) do dh, parts, dbc
    instructions = IgAShell.initial_upgrade_of_dofhandler(dh, igashell)
    Five.update_dofhandler!(dh, StateVariables(Float64, ndofs(dh)), instructions)
    
    alldofs = collect(1:ndofs(dh))
    JuAFEM.copy!!(dbc.free_dofs, alldofs)
end

solver = LocalDissipationSolver(
    Δλ0          = 10.0,
    Δλ_max       = 10.0,
    Δλ_min       = 1e-2,
    ΔL0          = 2.5,
    ΔL_min       = 1e-2,
    ΔL_max       = 5.0,
    sw2d         = 1.0,
    sw2i         = 1e-7,
    optitr       = 8,
    maxitr       = 50,
    maxsteps     = 1000,
    λ_max        = 700.0,
    λ_min        = -50.0,
    tol          = 1e-4,
    max_residual = 1e5
)




output = solvethis(solver, state, globaldata)


