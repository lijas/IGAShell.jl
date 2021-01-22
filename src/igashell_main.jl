export IGAShellState, IGAShell

"""
    IGAShell

Main IGAShell structure
"""
struct IGAShell{dim_p, dim_s, T, 
                data<:IGAShellData, 
                intdata<:IGAShellIntegrationData, 
                adapdata<:IGAShellAdaptivity, 
                vtkdata<:IGAShellVTK,
                srdata<:IGAShellStressRecovory} <: Five.AbstractPart{dim_s}

    layerdata::data
    integration_data::intdata
    adaptivity::adapdata
    vtkdata::vtkdata
    stress_recovory::srdata

    cellset::Vector{Int}
    cell_connectivity::Matrix{Int}
end

function IGAShell(;
    cellset      ::AbstractVector{Int}, 
    connectivity ::Matrix{Int},
    data         ::IGAShellData{dim_p,dim_s,T}
    ) where {dim_p,dim_s,T}

    _igashell_input_checks(data, cellset, connectivity)

    ncells = length(cellset)
    ncontrol_points = maximum(connectivity)

    #Setup adaptivity structure
    adapdata = IGAShellAdaptivity(data, connectivity, ncells, ncontrol_points)

    #
    Ce_mat, _ = IGA.compute_bezier_extraction_operators(data.orders[1:dim_p]..., data.knot_vectors[1:dim_p]...)
    Ce_vec = IGA.bezier_extraction_to_vectors(Ce_mat)
    
    #vtkdata
    vtkdata = IGAShellVTK(data)
    intdata = IGAShellIntegrationData(data, Ce_vec)
    srdata = IGAShellStressRecovory(data)

    return IGAShell{dim_p,dim_s,T, typeof(data), typeof(intdata), typeof(adapdata), typeof(vtkdata), typeof(srdata)}(data, intdata, adapdata, vtkdata, srdata, cellset, connectivity)

end

#Utility functions
layerdata(igashell::IGAShell) = igashell.layerdata
adapdata(igashell::IGAShell) = igashell.adaptivity
intdata(igashell::IGAShell) = igashell.integration_data
vtkdata(igashell::IGAShell) = igashell.vtkdata
srdata(igashell::IGAShell) = igashell.stress_recovory

cellconectivity!(nodes::Vector{Int}, igashell::IGAShell, cellid::Int) =  nodes .= igashell.cell_connectivity[:, cellid]
cellconectivity(igashell::IGAShell, cellid::Int) = @view igashell.cell_connectivity[:, cellid]

JuAFEM.getnquadpoints(igashell::IGAShell) = getnquadpoints_per_layer(igashell)*nlayers(igashell)
getnquadpoints_inplane(igashell::IGAShell) = length(getweights(intdata(igashell).iqr))
getnquadpoints_ooplane(igashell::IGAShell) = length(getweights(intdata(igashell).oqr))
getnquadpoints_per_interface(igashell::IGAShell{dim_p}) where {dim_p} = layerdata(igashell).nqp_interface_order^dim_p
getnquadpoints_ooplane_per_layer(igashell::IGAShell) = igashell.layerdata.nqp_ooplane_per_layer

getnquadpoints_per_layer(igashell::IGAShell) = igashell.layerdata.nqp_ooplane_per_layer * getnquadpoints_inplane(igashell)

nlayers(igashell::IGAShell) = igashell.layerdata.nlayers
ninterfaces(igashell::IGAShell) = nlayers(igashell)-1
ooplane_order(igashell::IGAShell{dim_p,dim_s}) where {dim_p,dim_s} = igashell.layerdata.orders[dim_s]

layer_thickness(igashell::IGAShell{dim_p,dim_s}, ilay::Int) where {dim_p,dim_s} = igashell.layerdata.zcoords[ilay+1] - igashell.layerdata.zcoords[ilay]

material(igashell::IGAShell) = igashell.layerdata.layer_materials
interface_material(igashell::IGAShell) = igashell.layerdata.interface_material

ndofs_per_controlpoint(igashell::IGAShell{dim_p,dim_s}, state::CELLSTATE) where {dim_p,dim_s} = ndofs_per_controlpoint(ooplane_order(igashell), nlayers(igashell), ninterfaces(igashell), dim_s, state)
ndofs_per_layer(igashell::IGAShell{dim_p,dim_s}) where {dim_p, dim_s} = (igashell.layerdata.orders[dim_s]+1) * JuAFEM.nnodes_per_cell(igashell) * dim_s


is_adaptive(igashell::IGAShell) = igashell.layerdata.adaptable

JuAFEM.nnodes_per_cell(igashell::IGAShell{dim_p}, cellid::Int=1) where dim_p = prod(igashell.layerdata.orders[1:dim_p].+1)::Int#getnbasefunctions(igashell.cv_inplane) ÷ dim_p
JuAFEM.getdim(igashell::IGAShell{dim_p,dim_s}) where {dim_p,dim_s} = dim_s
JuAFEM.getncells(igashell::IGAShell) = length(igashell.cellset)
JuAFEM.getnnodes(igashell::IGAShell) = maximum(igashell.cell_connectivity)

Five.get_fields(igashell::IGAShell) = [Field(:u, getmidsurface_ip(layerdata(igashell)), ndofs_per_controlpoint(igashell, LUMPED))]

Five.get_cellset(igashell::IGAShell) = igashell.cellset

get_inplane_qp_range(igashell::IGAShell; ilayer::Int, row::Int) = get_inplane_qp_range(getnquadpoints_per_layer(igashell), getnquadpoints_inplane(igashell), ilayer::Int, row::Int)
function get_inplane_qp_range(n_qp_per_layer::Int, n_inpqp::Int, ilayer::Int, row::Int)
    @assert( isless(row-1, n_qp_per_layer÷n_inpqp) )
    offset = (ilayer-1)*n_qp_per_layer + (row-1)*n_inpqp
    return (1:n_inpqp) .+ offset
end

function _igashell_input_checks(data::IGAShellData, cellset::AbstractVector{Int}, cell_connectivity::Matrix{Int})

    @assert(!any(is_mixed.(data.initial_cellstates)))

    #etc...
end

function Five.init_part!(igashell::IGAShell, dh::JuAFEM.AbstractDofHandler)
    _init_vtk_grid!(dh, igashell)
end

struct IGAShellState{MS1<:Five.AbstractMaterialState,MS2<:Five.AbstractMaterialState} <: Five.AbstractPartState
    materialstates::Array{MS1,2} #layer, qp
    interfacestates::Array{MS2,2} #interface, qp}
end

@inline getcellqps(state::IGAShellState, cellid::Int) = state.materialstates[cellid]
@inline getinterfaceqps(state::IGAShellState, cellid::Int) = state.interfacestates[cellid]

function Five.construct_partstates(igashell::IGAShell)
    
    cellmaterial = getmaterial(layerdata(igashell))[1]
    intmaterial = getinterfacematerial(layerdata(igashell))

    mstate = Five.get_material_state_type(cellmaterial)
    istate = Five.get_material_state_type(intmaterial)

    ninterface_qp = getnquadpoints_per_interface(igashell)
    nshell_qp = getnquadpoints_per_layer(igashell)

    states = Vector{IGAShellState{mstate,istate}}(undef,length(igashell.cellset))
    for ic in 1:length(igashell.cellset)
        interface_damage = layerdata(igashell).initial_interface_damages
        mstates = [Five.getmaterialstate(cellmaterial) for i in 1:nshell_qp, _ in 1:nlayers(igashell)]
        istates = [Five.getmaterialstate(intmaterial, interface_damage[iint, ic]) for _ in 1:ninterface_qp, iint in 1:ninterfaces(igashell)]

        states[ic] = IGAShellState{mstate, istate}(mstates, istates)
    end

    return states
end

function build_cellvalue!(igashell, cellid::Int)

    cellstate = getcellstate(igashell.adaptivity, cellid)
    
    if is_lumped(cellstate)
        cv =  intdata(igashell).cell_values_lumped
    elseif is_layered(cellstate)
        cv =  intdata(igashell).cell_values_layered
    elseif is_fully_discontiniuos(cellstate)
        cv =  intdata(igashell).cell_values_discont
    elseif is_discontiniuos(cellstate) || is_mixed(cellstate)
        cv = intdata(igashell).cell_values_mixed
        oop_values = _build_oop_basisvalue!(igashell, cellid)
        set_oop_basefunctions!(cv, oop_values)
    else
        error("wrong cellstate")
    end

    return cv
end 

function build_cohesive_cellvalue!(igashell, cellid::Int)
    cv_top = intdata(igashell).cell_values_cohesive_top
    cv_bot = intdata(igashell).cell_values_cohesive_bot

    oop_cohesive_top_values, oop_cohesive_bot_values = _build_oop_cohesive_basisvalue!(igashell, cellid)

    set_ooplane_basefunctions!(cv_top, oop_cohesive_top_values)
    set_ooplane_basefunctions!(cv_bot, oop_cohesive_bot_values)
    
    return cv_top, cv_bot
end

function build_facevalue!(igashell, faceidx::FaceIndex)
    cellid,faceid = faceidx

    cv = intdata(igashell).cell_values_face
    oop_values = _build_oop_basisvalue!(igashell, faceidx)
 
    set_quadraturerule!(cv, get_face_qr(intdata(igashell), faceid))
    set_ooplane_basefunctions!(cv, oop_values)

    return cv
end 

function build_facevalue!(igashell::IGAShell{1,2}, edgeidx::VertexIndex)
    cellid, vertexid = edgeidx
    
    vertexid = vertex_converter(igashell, vertexid)

    cv = intdata(igashell).cell_values_side
    
    oop_values = _build_oop_basisvalue!(igashell, cellid)
    basisvalues_inplane = cached_side_basisvalues(intdata(igashell), vertexid)

    set_inp_basefunctions!(cv, basisvalues_inplane)
    set_oop_basefunctions!(cv, oop_values)

    return cv
end 

function build_facevalue!(igashell, edgeidx::EdgeIndex)
    cellid, edgeid = edgeidx
    
    cv = intdata(igashell).cell_values_side
    
    oop_values = _build_oop_basisvalue!(igashell, cellid)
    basisvalues_inplane = cached_side_basisvalues(intdata(igashell), edgeid)

    set_inplane_basefunctions!(cv, basisvalues_inplane)
    set_ooplane_basefunctions!(cv, oop_values)

    return cv
end 

function build_facevalue!(igashell, edgeidx::EdgeInterfaceIndex)
    cellid, edgeid, face = edgeidx
   
    cv = intdata(igashell).cell_values_interface
    
    oop_values = _build_oop_basisvalue!(igashell, cellid)
    basisvalues_inplane = cached_side_basisvalues(intdata(igashell), edgeid)
    
    set_quadraturerule!(cv, get_face_qr(intdata(igashell), face))
    set_inplane_basefunctions!(cv, basisvalues_inplane)
    set_ooplane_basefunctions!(cv, oop_values) 

    return cv
end 

function build_facevalue!(igashell, vertex::VertexInterfaceIndex)
    cellid, vertexid, face = vertex
    
    vertexid = vertex_converter(igashell, vertexid)

    cv = intdata(igashell).cell_values_vertices
    
    oop_values = _build_oop_basisvalue!(igashell, FaceIndex(cellid, face))
    basisvalues_inplane = cached_vertex_basisvalues(intdata(igashell), vertexid)
    
    set_quadraturerule!(cv, get_face_qr(intdata(igashell), face))
    set_inplane_basefunctions!(cv, basisvalues_inplane)
    set_ooplane_basefunctions!(cv, oop_values) 

    return cv
end  

function build_active_layer_dofs(igashell::IGAShell, cellstate::CELLSTATE)

    if is_lumped(cellstate)
        return intdata(igashell).cache_values.active_layer_dofs_lumped
    elseif is_layered(cellstate)
        return intdata(igashell).cache_values.active_layer_dofs_layered
    elseif is_fully_discontiniuos(cellstate)
        return intdata(igashell).cache_values.active_layer_dofs_discont
    else
        return generate_active_layer_dofs(nlayers(igashell), ooplane_order(layerdata(igashell)), dim_s, states)
    end


end

function build_active_interface_dofs(igashell::IGAShell, cellstate::CELLSTATE)

    if is_lumped(cellstate)
        return intdata(igashell).cache_values.active_interface_dofs_lumped, intdata(igashell).cache_values.active_inplane_interface_dofs_lumped
    elseif is_layered(cellstate)
        return intdata(igashell).cache_values.active_interface_dofs_layered, intdata(igashell).cache_values.active_inplane_interface_dofs_layered
    elseif is_fully_discontiniuos(cellstate)
        return intdata(igashell).cache_values.active_interface_dofs_discont, intdata(igashell).cache_values.active_inplane_interface_dofs_discont
    else
        return generate_active_interface_dofs(ninterfaces(igashell), ooplane_order(layerdata(igashell)), dim_s, states)
    end

end

function _build_oop_basisvalue!(igashell::IGAShell{dim_p,dim_s,T}, idx::Union{Int, FaceIndex}) where {dim_p,dim_s,T}

    cellid = idx isa FaceIndex ? getcellid(idx) : idx

    cellnodes = zeros(Int, JuAFEM.nnodes_per_cell(igashell))
    cellconectivity!(cellnodes, igashell, cellid)

    oop_values = OOPBasisValues{T}[]

    for (i, nodeid) in enumerate(cellnodes)
        cp_state = get_controlpoint_state(adapdata(igashell), nodeid)
        
        if idx isa FaceIndex
            cv_oop = cached_face_basisvalues(intdata(igashell), cp_state, getidx(idx))
        else
            cv_oop = cached_cell_basisvalues(intdata(igashell), cp_state)
        end

        push!(oop_values, cv_oop)
    end
    return oop_values
end

function _build_oop_cohesive_basisvalue!(igashell::IGAShell{dim_p,dim_s,T}, cellid::Int) where {dim_p,dim_s,T}
    
    #Get cellconectivity
    cellnodes = zeros(Int, JuAFEM.nnodes_per_cell(igashell, cellid))
    cellconectivity = cellconectivity!(cellnodes, igashell, cellid)

    #..
    oop_cohesive_top_values = OOPBasisValues{T}[]
    oop_cohesive_bot_values = OOPBasisValues{T}[]

    for (i, nodeid) in enumerate(cellconectivity)
        cp_state = get_controlpoint_state(adapdata(igashell), nodeid)
        
        cached_bottom_values, cached_top_values = cached_cohesive_basisvalues(intdata(igashell), cp_state)
        
        push!(oop_cohesive_top_values, cached_top_values)
        push!(oop_cohesive_bot_values, cached_bottom_values)

    end

    return oop_cohesive_top_values, oop_cohesive_bot_values

end

@enum IGASHELL_ASSEMBLETYPE IGASHELL_FORCEVEC IGASHELL_STIFFMAT IGASHELL_FSTAR IGASHELL_DISSIPATION

function Five.assemble_fstar!(dh::JuAFEM.AbstractDofHandler, igashell::IGAShell, state::StateVariables)
    _assemble_stiffnessmatrix_and_forcevector!(dh, igashell, state, IGASHELL_FSTAR)
end

function Five.assemble_dissipation!(dh::JuAFEM.AbstractDofHandler, igashell::IGAShell, state::StateVariables)
    _assemble_stiffnessmatrix_and_forcevector!(dh, igashell, state, IGASHELL_DISSIPATION)
end


function Five.assemble_stiffnessmatrix_and_forcevector!( dh::JuAFEM.AbstractDofHandler, igashell::IGAShell, state::StateVariables) 
    _assemble_stiffnessmatrix_and_forcevector!(dh, igashell, state, IGASHELL_STIFFMAT)
end

function _assemble_stiffnessmatrix_and_forcevector!( dh::JuAFEM.AbstractDofHandler, 
                                                     igashell::IGAShell{dim_p,dim_s,T},  
                                                     state::StateVariables, 
                                                     assemtype::IGASHELL_ASSEMBLETYPE) where {dim_p,dim_s,T}

    assembler = start_assemble(state.system_arrays.Kⁱ, state.system_arrays.fⁱ, fillzero=false)  

    nnodes = JuAFEM.nnodes_per_cell(igashell)
    nlayerdofs = ndofs_per_layer(igashell)
    X = zeros(Vec{dim_s,T}, nnodes)
    Xᵇ = similar(X)
    layerdofs = zeros(Int, nlayerdofs)
    celldofs = zeros(Int, JuAFEM.ndofs_per_cell(dh, 1))
    fe = zeros(T, nlayerdofs)
    ke = zeros(T, nlayerdofs, nlayerdofs)
    ue = zeros(T, nlayerdofs)

    Δt = state.Δt

    V = 0
    @timeit "Shell loop" for (ic, cellid) in enumerate(igashell.cellset)
        cellstate = getcellstate(adapdata(igashell), ic)
            
        @timeit "buildcv" cv = build_cellvalue!(igashell, ic)
        active_layerdofs = build_active_layer_dofs(igashell, cellstate)

        C = get_extraction_operator(intdata(igashell), ic)

        IGA.set_bezier_operator!(cv, C)

        JuAFEM.cellcoords!(X, dh, cellid)
        Xᵇ .= IGA.compute_bezier_points(C, X)

        resize!(celldofs, JuAFEM.ndofs_per_cell(dh,cellid))
        JuAFEM.celldofs!(celldofs, dh, cellid)

        ⁿmaterialstates = state.prev_partstates[ic].materialstates
        materialstates = state.partstates[ic].materialstates

        @timeit "init_ms" reinit_midsurface!(cv, Xᵇ)
        for ilay in 1:nlayers(igashell)
            fill!(ke, 0.0); fill!(fe, 0.0)

            @timeit "init_lay" reinit_layer!(cv, ilay)

            layerdofs = celldofs[active_layerdofs[ilay]]
            
            ue .= state.d[layerdofs]

            ⁿstates =  @view ⁿmaterialstates[:, ilay]
            states = @view materialstates[:, ilay]

            #@timeit "integrate shell"
            V += _get_layer_forcevector_and_stiffnessmatrix!(
                                                cv, 
                                                ke, fe, 
                                                material(igashell)[ilay], states, ⁿstates, 
                                                ue, ilay, 
                                                is_small_deformation_theory(layerdata(igashell)), getwidth(layerdata(igashell)))

            assemble!(assembler, layerdofs, ke, fe)
        end
    end
    
    
    A = 0.0
    @timeit "Interface loop" for (ic, cellid) in enumerate(igashell.cellset)

        cellstate = getcellstate(adapdata(igashell), ic)

        if !is_discontiniuos(cellstate) && !is_mixed(cellstate)
            continue
        end

        Ce = get_extraction_operator(intdata(igashell), ic)

        cv_cohesive_top, cv_cohesive_bot = build_cohesive_cellvalue!(igashell, ic) 

        active_layerdofs = build_active_layer_dofs(igashell, cellstate) 
        active_interfacedofs, local_interface_dofs = build_active_interface_dofs(igashell, cellstate)

        ⁿinterfacestates = state.prev_partstates[ic].interfacestates
        interfacestates = state.partstates[ic].interfacestates

        resize!(celldofs, JuAFEM.ndofs_per_cell(dh,cellid))

        JuAFEM.cellcoords!(X, dh, cellid)
        JuAFEM.celldofs!(celldofs, dh, cellid)
        
        Xᵇ .= IGA.compute_bezier_points(Ce, X)
       
        IGA.set_bezier_operator!(cv_cohesive_top, Ce)
        IGA.set_bezier_operator!(cv_cohesive_bot, Ce)

        reinit_midsurface!(cv_cohesive_top, Xᵇ)
        reinit_midsurface!(cv_cohesive_bot, Xᵇ)
        for iint in 1:ninterfaces(igashell)      

            reinit_layer!(cv_cohesive_top, iint+1)
            reinit_layer!(cv_cohesive_bot, iint)
            
            ⁿstates =  @view ⁿinterfacestates[:, iint]
            states = @view interfacestates[:, iint]

            top_dofs = celldofs[active_layerdofs[iint+1]]
            bot_dofs = celldofs[active_layerdofs[iint]]
            
            ue_top = state.d[top_dofs]
            ue_bot = state.d[bot_dofs]

            fe_top = zeros(T, nlayerdofs)
            fe_bot = zeros(T, nlayerdofs)
            ke_top = zeros(T, nlayerdofs, nlayerdofs)  
            ke_bot = zeros(T, nlayerdofs, nlayerdofs)  

            if assemtype == IGASHELL_STIFFMAT
                @timeit "integrate_cohesive" A += integrate_cohesive_forcevector_and_stiffnessmatrix!(
                                                        cv_cohesive_top, cv_cohesive_bot,
                                                        interface_material(igashell), 
                                                        states,
                                                        ke_top, ke_bot,
                                                        fe_top, fe_bot,                
                                                        ue_top, ue_bot, 
                                                        Δt,
                                                        iint, getwidth(layerdata(igashell))) 

                assemble!(assembler, top_dofs, ke_top, fe_top)
                assemble!(assembler, bot_dofs, ke_bot, fe_bot)
                
            elseif assemtype == IGASHELL_FSTAR
                @timeit "integrate_cohesive_fstar" integrate_cohesive_fstar!(
                                                            cv_cohesive_top, cv_cohesive_bot,
                                                            interface_material(igashell), viscocity_parameter(layerdata(igashell)), 
                                                            ⁿstates, states,
                                                            ike, ife,                
                                                            ue_interface, 
                                                            Δue_interface, 
                                                            Δt,
                                                            iint, ninterfaces(igashell),
                                                            active_dofs, getwidth(layerdata(igashell))) 

                    state.system_arryas.fⁱ[celldofs[active_dofs]] += ife
            elseif assemtype == IGASHELL_DISSIPATION
                ge = Base.RefValue(zero(T))
                @timeit "integrate_cohesive_dissi" integrate_dissipation!(
                                                            cv_cohesive_top, cv_cohesive_bot,
                                                            interface_material(igashell), viscocity_parameter(layerdata(igashell)), 
                                                            ⁿstates, states,
                                                            ge, ife,                
                                                            ue_interface, 
                                                            Δue_interface, 
                                                            Δt,
                                                            iint, ninterfaces(igashell),
                                                            active_dofs, getwidth(layerdata(igashell))) 
                                                            
                state.system_arrays.G[] += ge[]
                state.system_arrays.fᴬ[celldofs[active_dofs]] += ife
            else
                error("wrong option")
            end
        end
    end 
end

function assemble_massmatrix!( dh::JuAFEM.AbstractDofHandler, igashell::IGAShell{dim_p,dim_s,T}, system_arrays::SystemArrays) where {dim_p,dim_s,T}


end

function Five.post_part!(dh, igashell::IGAShell{dim_p,dim_s,T}, states) where {dim_s, dim_p, T}
    #if dim_s == 2
    #    return
    #end
    return
    for (ic,cellid) in enumerate(igashell.cellset)#enumerate(CellIterator(dh, igashell.cellset))
        
        cellstate = getcellstate(adapdata(igashell), ic)

        if is_discontiniuos(cellstate)
            continue
        end

        #Get cellvalues for cell
        Ce = get_extraction_operator(intdata(igashell), ic)
        
        #Extract stresses from states
        σ_states = states.partstates[ic].materialstates[:]
        σ_states = getproperty.(σ_states, :σ)
        #Data for cell
        _celldofs = celldofs(dh, cellid)
        ue = states.d[_celldofs]

        nnodes = JuAFEM.nnodes_per_cell(igashell)
        X = zeros(Vec{dim_s,T}, nnodes)
        JuAFEM.cellcoords!(X, dh, cellid)
        Xᵇ= IGA.compute_bezier_points(Ce, X)
        celldata = (celldofs = _celldofs, 
                    Xᵇ=Xᵇ, X=X, ue=ue, 
                    nlayers=nlayers(igashell), ninterfaces=ninterfaces(igashell), 
                    cellid=cellid, ic=ic)

        #Build basis_values for cell
        cv = build_cellvalue!(igashell, ic)
        IGA.set_bezier_operator!(cv, Ce)
        reinit!(cv, Xᵇ)

        #Build basis_values for stress_recovory
        cv_sr = intdata(igashell).cell_values_sr
        oop_values = _build_oop_basisvalue!(igashell, ic, -1)
        set_oop_basefunctions!(cv_sr, oop_values)
        IGA.set_bezier_operator!(cv_sr, Ce)
        reinit!(cv_sr, Xᵇ)

        recover_cell_stresses(srdata(igashell), σ_states, celldata, cv_sr, cv)
    end

end

function _get_layer_forcevector_and_stiffnessmatrix!(
                                cv::IGAShellValues{dim_s,dim_p,T}, 
                                ke::AbstractMatrix, fe::AbstractVector,
                                layermat, materialstate, ⁿmaterialstate, 
                                ue::AbstractVector{T}, ilay::Int, 
                                is_small_deformation_theory::Bool, width::T) where {dim_s,dim_p,T}
                                
    ndofs_layer = getnbasefunctions_per_layer(cv)

    δF = zeros(Tensor{2,dim_s,T,dim_s^2}, ndofs_layer)
    δɛ = zeros(Tensor{2,dim_s,T,dim_s^2}, ndofs_layer)
    g = zeros(Vec{dim_s,T}, dim_s)

    qpᴸ = 0 #Counter for the layer qp
    V = 0.0
    for _ in 1:getnquadpoints_ooplane_per_layer(cv)
        for iqp in 1:getnquadpoints_inplane(cv)
            qpᴸ +=1 
            dΩ = getdetJdV(cv,qpᴸ) * width
            V += dΩ
            R = cv.R[qpᴸ]

            #Covarient triad in deformed configuration
            for d in 1:dim_s
                g[d] = cv.G[qpᴸ][d] + function_parent_derivative(cv, qpᴸ, ue, d)
            end
   
            # Deformation gradient
            F = zero(Tensor{2,dim_s,T})
            for i in 1:dim_s
                F += g[i]⊗cv.Gᴵ[qpᴸ][i]
            end
 
            # Variation of deformation gradient
            for i in 1:ndofs_layer
                _δF = zero(Tensor{2,dim_s,T})
                for d in 1:dim_s
                    δg = basis_parent_derivative(cv, qpᴸ, i, d)
                    _δF += δg ⊗ cv.Gᴵ[qpᴸ][d]
                end
                δF[i] = _δF
            end
            
            if is_small_deformation_theory
                _calculate_linear_forces!(fe, ke, cv, 
                                            ilay, qpᴸ, width,
                                            F, R, δF, δɛ, dΩ,
                                            layermat, materialstate, ⁿmaterialstate, 
                                            ndofs_layer)
            else
                _calculate_nonlinear_forces!(fe, ke, cv, 
                                            ilay, qpᴸ, width,
                                            F, R, δF, δɛ, dΩ,
                                            layermat, materialstate, ⁿmaterialstate, 
                                            ndofs_layer)
            end
        end
    end 
    return V

end

@inline function _calculate_linear_forces!(fe, ke, cv, ilay, layer_qp, width, F::Tensor{2,dim_s}, R, δF, δɛ, dΩ, layermat, materialstates, ⁿmaterialstates, ndofs_layer) where {dim_s}
    ɛ = symmetric(F) - one(SymmetricTensor{2,dim_s})
    
    δɛ .= symmetric.(δF)

    #Rotate strain
     _̂ε = symmetric(R' ⋅ ɛ ⋅ R)
     _̂σ, ∂̂σ∂ɛ, new_matstate = constitutive_driver(layermat, _̂ε, ⁿmaterialstates[layer_qp])
    materialstates[layer_qp] = new_matstate
    ∂σ∂ɛ = otimesu(R,R) ⊡ ∂̂σ∂ɛ ⊡ otimesu(R',R')
    σ = R⋅_̂σ⋅R'
    
    #σ, ∂σ∂ɛ, new_matstate = constitutive_driver(material[ilay], ɛ, ⁿmaterialstates[layer_qp])
    #materialstates[layer_qp] = new_matstate

    for i in 1:ndofs_layer

        δɛᵢ  = δɛ[i]
        
        fe[i] += (σ ⊡ δɛᵢ) * dΩ

        ɛC = δɛᵢ ⊡ ∂σ∂ɛ
        
        for j in 1:ndofs_layer
            
            δɛⱼ = δɛ[j]
            
            ke[i,j] += (ɛC ⊡ δɛⱼ) * dΩ
            
        end

    end

end

@inline function _calculate_nonlinear_forces!(fe, ke, cv, ilay, layer_qp, width, F, R, δF, δE, dΩ, layermat, materialstates, ⁿmaterialstates, ndofs_layer)

    E = symmetric(1/2 * (F' ⋅ F - one(F)))

    S, ∂S∂E, new_matstate = Five.constitutive_driver(layermat, E, ⁿmaterialstates[layer_qp])
    materialstates[layer_qp] = new_matstate

    σ = inv(det(F)) * F ⋅ S ⋅ F'

    # Hoist computations of δE
    for i in 1:ndofs_layer
        δFi = δF[i]
        δE[i] = symmetric(1/2*(δFi'⋅F + F'⋅δFi))
    end

    for i in 1:ndofs_layer
        δFi = δF[i]
        fe[i] += (δE[i] ⊡ S) * dΩ

        δE∂S∂E = δE[i] ⊡ ∂S∂E
        S∇δu = S ⋅ δFi'
        for j in 1:ndofs_layer
            δ∇uj = δF[j]
            ke[i, j] += (δE∂S∂E ⊡ δE[j] + S∇δu ⊡ δ∇uj' ) * dΩ
        end
    end

end

function integrate_cohesive_forcevector_and_stiffnessmatrix!(
    cv_top::IGAShellValues{dim_s,dim_p,T}, cv_bot::IGAShellValues{dim_s,dim_p,T},
    material::Five.AbstractMaterial, 
    materialstate::AbstractArray{<:Five.AbstractMaterialState}, 
    ke_top::AbstractMatrix, ke_bot::AbstractMatrix,
    fe_top::AbstractVector, fe_bot::AbstractVector,
    ue_top::AbstractVector, ue_bot::AbstractVector,
    Δt::T,
    iint::Int, active_basefunks::Vector{Int}, width::T,
    ) where {dim_s,dim_p,T}

    n_active_basefunction = length(active_basefunks)

    ndofs_layer = getnbasefunctions_per_layer(cv_top)
    A = 0.0

    for qp in 1:getnquadpoints_inplane(cv_top)
        
        #Rotation matrix
        R, dΓ = _cohesvive_rotation_matrix!(cv_top, cv_bot, qp, ue_top, ue_bot)
        dΓ *= width
        A += dΓ
        
        u₊ = function_value(cv_top, qp, ue_top); 
        u₋ = function_value(cv_bot, qp, ue_bot);
        J = u₊ - u₋
        Ĵ = R'⋅J
        
        
        #constitutive_driver
        t̂, ∂t∂Ĵ, new_matstate = Five.constitutive_driver(material, Ĵ, materialstate[qp])
        materialstate[qp] = new_matstate

        t = R⋅t̂
        ∂t∂J = R⋅∂t∂Ĵ⋅R'

        #top part
        for i in 1:n_active_basefunction
            Ni = cv_top.inplane_values_nurbs.N[active_basefunks[i],qp]
            for d in 1:dim
                #Bottom surface
                fe[i*dim - (dim-d)]       += -t[d] * Ni
                #Top surface
                fe[i*dim - (dim-d) + dim] += +t[d+1] * Ni
            end

            for j in 1:n_active_basefunction
                Nj = cv_top.inplane_values_nurbs.N[active_basefunks[j],qp]

                for d1 in 1:dim
                    for d2 in 1:dim
                        ii = i*dim - (dim-d)
                        jj = j*dim - (dim-d)
                        ke[ii, jj]             +=  -Nj * ∂t∂J[d1,d2] * -Ni
                        ke[ii + dim, jj]       +=  -Nj * ∂t∂J[d1,d2] * +Ni
                        ke[ii, jj + dim]       +=  +Nj * ∂t∂J[d1,d2] * -Ni
                        ke[ii + dim, jj + dim] +=  +Nj * ∂t∂J[d1,d2] * +Ni
                    end
                end
            end
        end
    end

    return A
end

function integrate_cohesive_fstar!(
    cv_top::IGAShellValues{dim_s,dim_p,T}, cv_bot,
    material::Five.AbstractMaterial, ξ::T,
    materialstate::AbstractArray{<:Five.AbstractMaterialState}, 
    new_materialstate::AbstractArray{<:Five.AbstractMaterialState}, 
    ke::AbstractMatrix, 
    fe::AbstractVector, 
    ue::AbstractVector,
    Δue::AbstractVector,
    Δt::T,
    iint::Int, ninterfaces::Int,
    active_dofs::AbstractVector{Int}, width::T,
    ) where {dim_s,dim_p,T}
    
    ndofs = length(active_dofs)

    qp_offset = (iint-1)*getnquadpoints_inplane(cv_top)
    for qp in (1:getnquadpoints_inplane(cv_top)) .+ qp_offset
        
        #Rotation matrix
        R, dΓ = _cohesvive_rotation_matrix!(cv_top, cv_bot, qp, ue, active_dofs)
        dΓ *= width
        
        u₊ = zero(Vec{dim_s,T}); u₋ = zero(Vec{dim_s,T}) 
        Δu₊ = zero(Vec{dim_s,T}); Δu₋ = zero(Vec{dim_s,T}) 
        for (i,j) in enumerate(active_dofs)
            u₊ += cv_top.U[j,qp] * ue[i]
            u₋ += cv_bot.U[j,qp] * ue[i]

            Δu₊ += cv_top.U[j,qp] * Δue[i]
            Δu₋ += cv_bot.U[j,qp] * Δue[i]
        end

        ΔJ = Δu₊ - Δu₋
        J = u₊ - u₋
        Ĵ = R'⋅J
        
        #constitutive_driver
        t̂, ∂t∂Ĵ, new_matstate = constitutive_driver(material, Ĵ, new_materialstate[qp-qp_offset])

        t = R⋅t̂
        ∂t∂J = R⋅∂t∂Ĵ⋅R'

        #Add viscocity term
        #K = 1.0 #initial_stiffness(material)
        #σᵛ = ξ *K .* ΔJ/Δt
        #∂σᵛ∂J = ξ * K/Δt * one(SymmetricTensor{2,dim_s,T})

        #∂t∂J += ∂σᵛ∂J
        #t += σᵛ

        #if iszero(t̂)
        #    continue
        #end

        for i in 1:ndofs
            J̇ = basis_value(cv_top, qp, active_dofs[i]) - basis_value(cv_bot, qp, active_dofs[i])
            fe[i] += (J ⋅ ∂t∂J ⋅ J̇) * dΓ

        end
    end

end

function integrate_dissipation!(
    cv_top::IGAShellValues{dim_s,dim_p,T}, cv_bot,
    material::Five.AbstractMaterial, ξ::T,
    materialstate::AbstractArray{<:Five.AbstractMaterialState}, 
    new_materialstate::AbstractArray{<:Five.AbstractMaterialState}, 
    ge::Base.RefValue, 
    fe::AbstractVector, 
    ue::AbstractVector,
    Δue::AbstractVector,
    Δt::T,
    iint::Int, ninterfaces::Int,
    active_dofs::AbstractVector{Int}, width::T,
    ) where {dim_s,dim_p,T}
    
    ndofs = length(active_dofs)

    A = 0.0
    
    qp_offset = (iint-1)*getnquadpoints_inplane(cv_top)
    for qp in (1:getnquadpoints_inplane(cv_top)) .+ qp_offset
        
        #Rotation matrix
        R, dΓ = _cohesvive_rotation_matrix!(cv_top, cv_bot, qp, ue, active_dofs)
        dΓ *= width
        
        u₊ = zero(Vec{dim_s,T}); u₋ = zero(Vec{dim_s,T}) 
        for (i,j) in enumerate(active_dofs)
            u₊ += cv_top.N[j,qp] * ue[i]
            u₋ += cv_bot.N[j,qp] * ue[i]
        end

        J = u₊ - u₋
        Ĵ = R'⋅J

        # The constitutive_driver calucaleted the dissipation and dgdJ in the internal force integration loop
        # and stored it in the state variable...
        g, dgdĴ = Five.constitutive_driver_dissipation(material, Ĵ, new_materialstate[qp-qp_offset])
        dgdJ =  R ⋅ dgdĴ

        ge[] += g * dΓ
        for i in 1:ndofs
            δui = basis_value(cv_top, qp, active_dofs[i]) - basis_value(cv_bot, qp, active_dofs[i])
            fe[i] += (dgdJ ⋅ δui) * dΓ
        end
    end

    return A
end

function _cohesvive_rotation_matrix!(cv_top::IGAShellValues{dim_s,dim_p,T}, 
                                     cv_bot::IGAShellValues{dim_s,dim_p,T}, 
                                     qp::Int,
                                     ue_top::AbstractVector{T}, ue_bot::AbstractVector{T}) where {dim_p,dim_s,T}


    g₊ = zeros(Vec{dim_s,T},dim_p)
    g₋ = zeros(Vec{dim_s,T},dim_p)
    
    for d in 1:dim_p
        g₊[d] = cv_top.G[qp][d] + function_parent_derivative(cv_top, qp, ue_top, d)
        g₋[d] = cv_bot.G[qp][d] + function_parent_derivative(cv_bot, qp, ue_bot, d)
    end
    
    local _R, detJ
    if dim_s == 3
        n₊ = cross(g₊[1], g₊[2])
        n₊ /= norm(n₊)
        n₋ = cross(g₋[1], g₋[2])
        n₋ /= norm(n₋)
        n = 0.5(n₊ + n₋)

        s₊₂ = g₊[1]/norm(g₊[1])
        s₋₂ = g₋[1]/norm(g₋[1])
        s₂ = 0.5(s₊₂ + s₋₂)

        s₊₃ = cross(n₊, s₊₂)
        s₋₃ = cross(n₋, s₋₂)
        s₃ = 0.5(s₊₃ + s₋₃)

        #=
        e₁, e₂, e₃ = basevec(Vec{dim_s,T})
        _cos(e,a) = dot(e,a)/norm(e)/norm(a)
        _R = (_cos(e₁,s₂), _cos(e₂,s₂), _cos(e₃,s₂), 
            _cos(e₁,s₃), _cos(e₂,s₃), _cos(e₃,s₃),
            _cos(e₁,n),  _cos(e₂,n),  _cos(e₃,n))
        =#
        
        _cos(i,a) = a[i]/norm(a)
        _R = (_cos(1,s₂), _cos(2,s₂), _cos(3,s₂), 
            _cos(1,s₃), _cos(2,s₃), _cos(3,s₃),
            _cos(1,n ), _cos(2,n ), _cos(3,n))

        detJ = norm(cross(0.5*(g₊[1] + g₋[1]),
                    0.5*(g₊[2] + g₋[2])))
    else dim_s == 2
        n₊ = Vec{2,T}((-g₊[1][2], g₊[1][1]))
        n₊ /= norm(n₊)
        n₋ = Vec{2,T}((-g₋[1][2], g₋[1][1]))
        n₋ /= norm(n₋)
        n = 0.5(n₊ + n₋)

        s₊₂ = g₊[1]/norm(g₊[1])
        s₋₂ = g₋[1]/norm(g₋[1])
        s₂ = 0.5(s₊₂ + s₋₂)

        #_R = (s₂[1], s₂[2], n[1], n[2])
        _cos2d(i,a) = a[i]/norm(a)
        _R = (_cos2d(1,s₂), _cos2d(2,s₂), _cos2d(1,n), _cos2d(2,n))
        detJ = norm(0.5*(g₊[1] + g₋[1]))
        
    end
    
    R = Tensor{2,dim_s,T,dim_s^2}(_R)

    @assert( det(R)≈1.0 )

    dV = detJ * get_qp_weight(cv_top, qp)
    
    return R, dV
end

function Five.get_vtk_grid(dh::JuAFEM.AbstractDofHandler, igashell::IGAShell{dim_p,dim_s,T}) where {dim_p,dim_s,T}
    return igashell.vtkdata.cls, igashell.vtkdata.node_coords
end

function Five.commit_part!(dh::JuAFEM.AbstractDofHandler, igashell::IGAShell{dim_p,dim_s}, state::StateVariables) where {dim_p,dim_s}
    
    if !is_adaptive(igashell)
        return FieldDimUpgradeInstruction[]
    end
    
    _commit_part!(dh, igashell, state)
end
