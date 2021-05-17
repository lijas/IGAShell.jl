using Five
using IgAShell

#Dimension
DIM = 2
NELX = 175
NELY = 1

ORDERS = (2,3)

L = 120.0
h = 4.0
b = 20.0
a0 = 46.9

angles = deg2rad.([0.0, 0.0])
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
    Gᴵ   = (0.5, 0.5, 0.5),
    τᴹᵃˣ = (50.0, 50.0, 50.0),
    η    = 1.6
) 

material(_α) = 
MatTransvLinearElastic(
    E1 = 126.0e3,
    E2 = 10.0e3,
    ν_12 = 0.29, 
    G_12 = 8.0e3, 
    α = _α
) 
layermats = [Material2D(material(α), Five.PLANE_STRESS) for α in angles]

#
nurbsmesh = IgAShell.IGA.generate_nurbs_patch(:line, (NELX, ), (ORDERS[1], ), (L, ), sdim=2) 
data.grid = IgAShell.IGA.Grid(nurbsmesh)

#Sets
addcellset!(data.grid, "precracked", (x) -> x[1] > L-a0)
precracked_cells = collect(getcellset(data.grid, "precracked"))
addvertexset!(data.grid, "right", (x)-> x[1] ≈ L)
addvertexset!(data.grid, "left", (x)-> x[1] ≈ 0.0)
addvertexset!(data.grid, "mid", (x)-> x[1] ≈ L/2)
addcellset!(data.grid, "topcells", (x)-> true)
@assert(  !isempty(getvertexset(data.grid, "mid"))  )
partset1 = collect(1:length(data.grid.cells))

#
cellstates = [IgAShell.LAYERED for i in 1:NELX]
cellstates[precracked_cells] .= IgAShell.FULLY_DISCONTINIUOS

interface_damage = zeros(nlayers-1, getncells(data.grid))
interface_damage[1, precracked_cells] .= 1.0

#IGAshell data
igashelldata = 
IgAShell.IGAShellData(;
    layer_materials           = layermats,
    interface_material        = interfacematerial,
    orders                    = ORDERS,
    knot_vectors              = nurbsmesh.knot_vectors,
    thickness                 = h,
    width                     = DIM == 2 ? b : 1.0,
    initial_cellstates        = cellstates,
    initial_interface_damages = interface_damage,
    nqp_inplane_order         = 3,
    nqp_ooplane_per_layer     = 4,
    adaptable                 = true,
        limit_stress_criterion   = 0.5,
        limit_damage_criterion   = 0.01,
        search_radius            = 10.0,
    small_deformations_theory = false,
    nqp_interface_order       = 4
)  

igashell = 
IgAShell.IGAShell(
    cellset = partset1, 
    data = igashelldata
) 
push!(data.parts, igashell)

#
data.output[] = Output(
    interval = 0.0,
    runname = "enf_2d",
    savepath = "./"
)

#
edgeset = VertexInterfaceIndex(getvertexset(data.grid, "left"), 1)
etf = IGAShellWeakBC( 
    set = edgeset,
    func = (x,t) -> zeros(DIM), 
    comps = 1:DIM,
    igashell = igashell, 
    penalty = 1e8
)
push!(data.constraints, etf)

#
edgeset = VertexInterfaceIndex(getvertexset(data.grid, "right"), 1)
etf = IGAShellWeakBC( 
    set = edgeset,
    func = (x,t) -> [0.0], 
    comps = [DIM],
    igashell = igashell, 
    penalty = 1e8
)
push!(data.constraints, etf)

#Force
midvertex = collect(getvertexset(data.grid, "mid"))[2]
edgeset = VertexInterfaceIndex([midvertex], 2)
etf = IGAShellExternalForce(
    set = edgeset, 
    func = (x,t) -> [0.0, -1.0/b],
    igashell = igashell
)
push!(data.external_forces, etf)

#
output = OutputData(
    type = IGAShellBCOutput(
        igashell = igashell,
        comps = [DIM]
    ),
    interval = 0.0,
    set      = edgeset
)
data.outputdata["reactionforce"] = output

#
vtkoutput = VTKCellOutput(
    type = IGAShellConfigStateOutput()
)
Five.push_vtkoutput!(data.output[], vtkoutput)

#
vtkoutput = VTKNodeOutput(
    type = IGAShellMaterialStateOutput(
        field = :interface_damage,
        dir = 2
    ),
)
Five.push_vtkoutput!(data.output[], vtkoutput)


#
output = OutputData(
    type = IGAShellStressOutput(
        igashell = igashell,
    ),
    interval = 0.0,
    set      = [90, 100, 107, 108, 109, 110, 111, 150]
)
data.outputdata["Stress at 50%"] = output

#
output = OutputData(
    type = IGAShellRecovoredStressOutput(
        igashell = igashell,
    ),
    interval = 0.0,
    set      = [90, 100, 107, 108, 109, 110, 111, 150]
)
data.outputdata["RS at 50%"] = output

state, globaldata = build_problem(data) do dh, parts, dbc
    instructions = IgAShell.initial_upgrade_of_dofhandler(dh, igashell)
    Five.update_dofhandler!(dh, StateVariables(Float64, ndofs(dh)), instructions)
    
    alldofs = collect(1:ndofs(dh))
    Ferrite.copy!!(dbc.free_dofs, alldofs)
end

solver = LocalDissipationSolver(
    Δλ0          = 5.0,
    Δλ_max       = 20.0,
    Δλ_min       = 1e-2,
    ΔL0          = 2.5,
    ΔL_min       = 1e-2,
    ΔL_max       = 5.0,
    sw2d         = 0.5,
    sw2i         = 1e-7,
    optitr       = 8,
    maxitr       = 50,
    maxsteps     = 5,
    λ_max        = 400.0,
    λ_min        = -100.0,
    tol          = 1e-4,
    max_residual = 1e5
)


output = solvethis(solver, state, globaldata)

d = [output.outputdata["reactionforce"].data[i].displacements for i in 1:length(output.outputdata["reactionforce"].data)]

using Test
@test all( d .≈ [0.0, 0.023551100017999857, 0.047134360028778355, 0.07823472445107986, 0.11927576443187565, 0.17343622364288683, 0.24012521130370895])

