using Five
using IgAShell

#Dimension
DIM = 3
NELX = 50
NELY = 3
ORDERS = (3,3,2)

#Geometry
P = 1.0
L = 10.0
h = 0.1
b = 1.0

#
angles = deg2rad.([0.0, 0.0])

data = ProblemData(
    dim = DIM,
    tend = 1.0,
    adaptive = true
)

interfacematerial = MatCZBilinear(
    K    = 1.0e5,
    Gᴵ   = (0.5, 0.5, 0.5),
    τᴹᵃˣ = (50.0, 50.0, 50.0),
    η    = 1.6
) 

material(_α) = 
MatLinearElastic(
    E = 200e5,
    nu = 0.0
) 

layermats = [material(α) for α in angles]

#
nurbsmesh = IgAShell.IGA.generate_nurbsmesh((NELX, NELY), ORDERS[1:2], (L, b), sdim=DIM) 
data.grid = IgAShell.IGA.convert_to_grid_representation(nurbsmesh)

#Setsk and variables
nlayers = length(angles)
ninterfaces = nlayers - 1
addedgeset!(data.grid, "left", (x)-> x[1] ≈ 0.0)
topfaceset = [FaceIndex(cellid, 2) for cellid in 1:getncells(data.grid)]
addedgeset!(data.grid, "tip", (x)->x[1] ≈ L)
partset1 = collect(1:length(data.grid.cells))

#
cellstates = [IgAShell.LUMPED for i in 1:getncells(data.grid)]
interface_damage = zeros(ninterfaces, getncells(data.grid))

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
    LIMIT_UPGRADE_INTERFACE   = 0.04,
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
    runname = "cantilever_beam",
    savepath = "./"
)

#
etf = IGAShellWeakBC( 
    set = getedgeset(data.grid, "left"),
    func = (x,t) -> zeros(DIM), 
    comps = 1:DIM,
    igashell = igashell, 
    penalty = 1e12
)
push!(data.constraints, etf)

#Force
etf = IGAShellExternalForce(
    set = topfaceset, 
    func = (x,t) -> Vec{DIM}((0.0, 0.0, t* -P/(b*L))),
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
    set      = getedgeset(data.grid, "tip")
)
data.outputdata["maxdisp"] = output

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


#Stress output
postcells = [50, 75, 90]
stress_output = IGAShell.IGAShellStressOutput(Ref(igashell), cellset = postcells, interval = 0.00)
data.outputs["Stress at 50%"] = stress_output

stress_output = IGAShell.IGAShellRecovoredStressOutput(Ref(igashell), cellset = postcells, interval = 0.00)
data.outputs["RS at 50%"] = stress_output
state, globaldata = build_problem(data) do dh, parts, dbc
    instructions = IgAShell.initial_upgrade_of_dofhandler(dh, igashell)
    Five.update_dofhandler!(dh, StateVariables(Float64, ndofs(dh)), instructions)
    
    alldofs = collect(1:ndofs(dh))
    JuAFEM.copy!!(dbc.free_dofs, alldofs)
end

solver = NewtonSolver(
    Δt0 = 1.0, 
    Δt_max = 1.0, 
    tol=1e-5
)


output = solvethis(solver, state, globaldata)

if true
    d = [output.outputdata["maxdisp"].data[i].displacements for i in 1:length(output.outputdata["maxdisp"].data)]
    f = [output.outputdata["maxdisp"].data[i].forces for i in 1:length(output.outputdata["maxdisp"].data)]
end



