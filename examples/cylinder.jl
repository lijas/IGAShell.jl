using Five
using IgAShell

#Dimension
DIM = 3
NELX = 15
NELY = 15
ORDERS = (3,3,2)
@assert(isodd(NELY) && isodd(NELX)) #So outputed cell for stresses is in the center

#Geometry
F = 100
h = 0.04
R = 10.0

#
angles = deg2rad.([0.0])

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
    E = 6.825e7,
    nu = 0.3
) 

layermats = [material(α) for α in angles]

#
nurbsmesh = IgAShell.IGA.generate_cylinder((NELX,NELY), ORDERS[1:2], r = 4.0, α = 1*pi, h = 10.0, twist_angle = 0*pi/4, multiplicity=(1,1))
data.grid = IgAShell.IGA.convert_to_grid_representation(nurbsmesh)

#Setsk and variables
nlayers = length(angles)
ninterfaces = nlayers - 1
addedgeset!(data.grid, "bot", (x) -> x[3] ≈ 0.0)
addedgeset!(data.grid, "top", (x) -> x[3] ≈ 10.0)
partset1 = collect(1:(NELX*NELY))
postcell = ceil(Int, 0.5*NELX*NELY)

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
    initial_cellstates        = cellstates,
    initial_interface_damages = interface_damage,
    nqp_inplane_order         = 4,
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
    runname = string("cylinder"),
    savepath = "./"
)

#
# BC
#
etf = IGAShellWeakBC( 
    set = getedgeset(data.grid, "bot"),
    func = (x,t) -> (0.0,0.0,0.0), 
    comps = [1,2,3],
    igashell = igashell, 
    penalty = 1e10
)
push!(data.constraints, etf)


#
#Force
#
etf = IGAShellExternalForce(
    set = getedgeset(data.grid, "top"),
    func = (x,t) -> [0*t*F, 0.0, 0.0],
    igashell = igashell
)
push!(data.external_forces, etf)


#
#
#


#=output = OutputData(
    type = IGAShellStressOutput(
        igashell = igashell,
    ),
    interval = 0.0,
    set      = [postcell]
)
data.outputdata["Stress at 50%"] = output

#
output = OutputData(
    type = IGAShellRecovoredStressOutput(
        igashell = igashell,
    ),
    interval = 0.0,
    set      = [postcell]
)
data.outputdata["RS at 50%"] = output=#

#
state, globaldata = build_problem(data) do dh, parts, dbc
    instructions = IgAShell.initial_upgrade_of_dofhandler(dh, igashell)
    Five.update_dofhandler!(dh, StateVariables(Float64, ndofs(dh)), instructions)
    
    alldofs = collect(1:ndofs(dh))
    JuAFEM.copy!!(dbc.free_dofs, alldofs)
end

solver = NewtonSolver(
    Δt0 = 0.1, 
    Δt_max = 0.1, 
    tol=1e-5,
    maxitr = 10
)


output = solvethis(solver, state, globaldata)

if false
    using Plots; pyplot(); PyPlot.pygui(true)
    using JLD2; using FileIO

    #
    dA = [output.outputdata["pointA"].data[i].displacements for i in 1:length(output.outputdata["pointA"].data)]
    t = [0.0, 0.05, 0.15000000000000002, 0.19204482076268575, 0.2025560259533572, 0.20781162854869292, 0.21002133723990088, 0.21133523788873482, 0.21211648788873483, 0.21277343821315178, 0.21332586538595377, 0.21387829255875576, 0.21453524288317272, 0.2154643109417686, 0.21656916528737258, 0.21788306593620652, 0.2194455659362065, 0.22130370205339825, 0.2235134107446062, 0.22614121204227405, 0.2287690133399419, 0.23139681463760975, 0.23452181463760974, 0.23823808687199324, 0.24195435910637675, 0.24567063134076025, 0.2500900487231762, 0.2553456513185119, 0.2606012539138476, 0.2658568565091833, 0.27210685650918326, 0.27953940097795027, 0.2869719454467173, 0.2944044899154843, 0.3032433246803161, 0.31375452987098756, 0.3225933646358194, 0.3314321994006512, 0.34027103416548304, 0.3507822393561545, 0.3632822393561545, 0.37379344454682595, 0.3843046497374974, 0.3931434845023292, 0.40365468969300067, 0.4124935244578325, 0.4213323592226643, 0.43017119398749615, 0.4406823991781676, 0.4531823991781676, 0.4656823991781676, 0.4781823991781676, 0.49304748811570165, 0.5107251576453653, 0.5255902465828993, 0.5404553355204333, 0.5529553355204333, 0.5678204244579673, 0.5803204244579673, 0.5908316296486387, 0.6013428348393101, 0.6118540400299814, 0.6243540400299814, 0.6392191289675154, 0.6540842179050494, 0.6689493068425835, 0.6866269763722471, 0.7076493867535899, 0.7253270562832536, 0.7430047258129172, 0.7578698147504512, 0.7755474842801149, 0.7904125732176489, 0.8029125732176489, 0.8154125732176488, 0.8279125732176488, 0.8427776621551828, 0.8604553316848464, 0.8781330012145101, 0.8958106707441738, 0.9168330811255165, 0.9418330811255164, 0.9628554915068592, 0.983877901888202, 1.0015555714178657]
    fA = 100 * t

    #
    dB = [output.outputdata["pointsB"].data[i].displacements for i in 1:length(output.outputdata["pointsB"].data)]
    t = [0.0, 0.05, 0.15000000000000002, 0.19204482076268575, 0.2025560259533572, 0.20781162854869292, 0.21002133723990088, 0.21133523788873482, 0.21211648788873483, 0.21277343821315178, 0.21332586538595377, 0.21387829255875576, 0.21453524288317272, 0.2154643109417686, 0.21656916528737258, 0.21788306593620652, 0.2194455659362065, 0.22130370205339825, 0.2235134107446062, 0.22614121204227405, 0.2287690133399419, 0.23139681463760975, 0.23452181463760974, 0.23823808687199324, 0.24195435910637675, 0.24567063134076025, 0.2500900487231762, 0.2553456513185119, 0.2606012539138476, 0.2658568565091833, 0.27210685650918326, 0.27953940097795027, 0.2869719454467173, 0.2944044899154843, 0.3032433246803161, 0.31375452987098756, 0.3225933646358194, 0.3314321994006512, 0.34027103416548304, 0.3507822393561545, 0.3632822393561545, 0.37379344454682595, 0.3843046497374974, 0.3931434845023292, 0.40365468969300067, 0.4124935244578325, 0.4213323592226643, 0.43017119398749615, 0.4406823991781676, 0.4531823991781676, 0.4656823991781676, 0.4781823991781676, 0.49304748811570165, 0.5107251576453653, 0.5255902465828993, 0.5404553355204333, 0.5529553355204333, 0.5678204244579673, 0.5803204244579673, 0.5908316296486387, 0.6013428348393101, 0.6118540400299814, 0.6243540400299814, 0.6392191289675154, 0.6540842179050494, 0.6689493068425835, 0.6866269763722471, 0.7076493867535899, 0.7253270562832536, 0.7430047258129172, 0.7578698147504512, 0.7755474842801149, 0.7904125732176489, 0.8029125732176489, 0.8154125732176488, 0.8279125732176488, 0.8427776621551828, 0.8604553316848464, 0.8781330012145101, 0.8958106707441738, 0.9168330811255165, 0.9418330811255164, 0.9628554915068592, 0.983877901888202, 1.0015555714178657]
    fB = 100 * t

    #Refereence
    ref_d_b = [0.0,0.854,1.459,1.939,2.254,2.536,2.735,2.901,3.05,3.166,3.273]
    ref_f_b = [0.0,10.497,20.718,30.387,40.331,50.276,60.221,70.442,79.834,90.055,99.724]

    ref_d_a = [0.0,0.97,1.773,2.519,3.149,3.746,4.227,4.641,4.989,5.354,5.66]
    ref_f_a = [0.0,10.221,20.442,30.11,40.055,50.276,59.945,69.337,79.282,89.779,99.448]

    plot()
    plot!(ref_d_a, ref_f_a, label = "Reference A")
    plot!(ref_d_b, ref_f_b, label = "Reference B")

    plot!(dA,fA, label = "point a")
    plot!(dB,fB, label = "point b")

end



