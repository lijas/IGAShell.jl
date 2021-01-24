using Five
using IgAShell

#Dimension
const DIM = 2
const NELX = 175
const NELY = 1

const ORDERS = (2,2)

const L = 120.0
const h = 4.0
const b = 20.0
const a0 = 46.9

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
    η    = 1.0
) 

material(_α) = 
MatTransvLinearElastic(
    E1 = 126.0e3,
    E2 = 10.0e3,
    ν_12 = 0.29, 
    G_12 = 8.0e3, 
    α = _α
) 
layermats = [Material2D(material(α), Five.PLANE_STRAIN) for α in angles]

#
nurbsmesh = IgAShell.IGA.generate_nurbsmesh((NELX, ), (ORDERS[1], ), (L, ), sdim=DIM) 
data.grid = IgAShell.IGA.convert_to_grid_representation(nurbsmesh)

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
cellstates = [IgAShell.FULLY_DISCONTINIUOS for i in 1:NELX]
#cellstates[precracked_cells] .= eliasfem.FULLY_DISCONTINIUOS

interface_damage = zeros(nlayers-1, getncells(data.grid))
interface_damage[1, precracked_cells] .= 1.0

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
    LIMIT_UPGRADE_INTERFACE   = 0.01,
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
midvertex = collect(getvertexset(data.grid, "mid"))[1]
edgeset = VertexInterfaceIndex([midvertex], 2)
etf = IGAShellExternalForce(
    set = edgeset, 
    func = (x,t) -> [0.0, -1.0/b],
    igashell = igashell
)
push!(data.external_forces, etf)

#
output = OutputData(
    type = IgAShell.IGAShellBCOutput(
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
    type = MaterialStateOutput(
        field = :interface_damage
    ),
)
Five.push_vtkoutput!(data.output[], vtkoutput)

#=
#Stress output
postcells = [50, 75, 90]
stress_output = IGAShell.IGAShellStressOutput(Ref(igashell), cellset = postcells, interval = 0.00)
data.outputs["Stress at 50%"] = stress_output

stress_output = IGAShell.IGAShellRecovoredStressOutput(Ref(igashell), cellset = postcells, interval = 0.00)
data.outputs["RS at 50%"] = stress_output=#

#=solver = NewtonSolver(
    Δt0 = 0.1,
    Δt_max = 0.1,

)
=#

state, globaldata = build_problem(data) do dh, parts, dbc
    instructions = IgAShell.initial_upgrade_of_dofhandler(dh, igashell)
    Five.update_dofhandler!(dh, StateVariables(Float64, ndofs(dh)), instructions)
    
    alldofs = collect(1:ndofs(dh))
    JuAFEM.copy!!(dbc.free_dofs, alldofs)
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
    maxsteps     = 200,
    λ_max        = 400.0*2,
    λ_min        = -100.0,
    tol          = 1e-4,
    max_residual = 1e5
)

#=solver = NewtonSolver(
    Δt0 = 0.1,
    Δt_max = 0.1,
    maxitr_first_step = 50
)=#


output = solvethis(solver, state, globaldata)

d = [output.outputdata["reactionforce"].data[i].displacements for i in 1:length(output.outputdata["reactionforce"].data)]
f = [output.outputdata["reactionforce"].data[i].forces for i in 1:length(output.outputdata["reactionforce"].data)]


fig = plot(reuse=false)
E₁₁ = 126.0e3

Gc = 0.5; 
I = 1/12 * b * (h/2)^3; 

uz1(F) = F*(L^3 + 12a0^3)/(384*E₁₁ * I)
uz2(F) = F*(L^3)/(384*E₁₁ * I) + 16 ./F.^2 * sqrt(E₁₁*I) * (b*Gc/3)^(3/2)
plot!(fig, abs.(d), abs.(f), label="adaptiv")
plot!(fig, uz1(0.0:0.1:450), 0.0:0.1:450, linestyle = :dashed, c="black")
plot!(fig, uz2(450:-0.1:250.0), 450:-0.1:250.0, linestyle = :dashed, c="black")
plot!(fig, ylimit = [0,500], yticks = 0.0:100.0:500.0)
display(fig)

