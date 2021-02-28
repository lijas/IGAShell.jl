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

interface_material(igashell::IGAShell) = igashell.layerdata.interface_material

ndofs_per_controlpoint(igashell::IGAShell{dim_p,dim_s}, state::CPSTATE) where {dim_p,dim_s} = ndofs_per_controlpoint(ooplane_order(igashell), 
                                                                                                    nlayers(igashell),
                                                                                                    ninterfaces(igashell),
                                                                                                    dim_s,
                                                                                                    state)

is_adaptive(igashell::IGAShell) = igashell.layerdata.adaptable

getcellstate(igashell::IGAShell, i::Int) = igashell.adaptivity.cellstates[i]

JuAFEM.nnodes_per_cell(igashell::IGAShell{dim_p}, cellid::Int=1) where dim_p = prod(igashell.layerdata.orders[1:dim_p].+1)::Int#getnbasefunctions(igashell.cv_inplane) ÷ dim_p
JuAFEM.getdim(igashell::IGAShell{dim_p,dim_s}) where {dim_p,dim_s} = dim_s
JuAFEM.getncells(igashell::IGAShell) = length(igashell.cellset)
JuAFEM.getnnodes(igashell::IGAShell) = maximum(igashell.cell_connectivity)

Five.get_fields(igashell::IGAShell) = [Field(:u, getmidsurface_ip(layerdata(igashell)), ndofs_per_controlpoint(igashell, LUMPED_CPSTATE))]

Five.get_cellset(igashell::IGAShell) = igashell.cellset

get_inplane_qp_range(igashell::IGAShell; ilayer::Int, row::Int) = get_inplane_qp_range(getnquadpoints_per_layer(igashell), getnquadpoints_inplane(igashell), ilayer::Int, row::Int)
function get_inplane_qp_range(n_qp_per_layer::Int, n_inpqp::Int, ilayer::Int, row::Int)
    @assert( isless(row-1, n_qp_per_layer÷n_inpqp) )
    offset = (ilayer-1)*n_qp_per_layer + (row-1)*n_inpqp
    return (1:n_inpqp) .+ offset
end

function _igashell_input_checks(data::IGAShellData{dim_p, dim_s}, cellset::AbstractVector{Int}, cell_connectivity::Matrix{Int}) where {dim_s,dim_p}

    @assert(!any(is_mixed.(data.initial_cellstates)))
    @assert( dim_s == length(data.orders) )
    #etc...
end

function IGAShell(;
    cellset::AbstractVector{Int}, 
    connectivity::Matrix{Int},
    data::IGAShellData{dim_p,dim_s,T}
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

function build_cellvalue!(igashell, cellstate::CELLSTATE)

    local cv
    if is_lumped(cellstate)
        cv =  intdata(igashell).cell_values_lumped
    elseif is_layered(cellstate)
        cv =  intdata(igashell).cell_values_layered
    elseif is_fully_discontiniuos(cellstate)
        cv =  intdata(igashell).cell_values_discont
    elseif has_discontinuity(cellstate) || is_mixed(cellstate)
        cv = intdata(igashell).cell_values_mixed
        oop_values = _build_oop_basisvalue!(igashell, cellstate)
        set_oop_basefunctions!(cv, oop_values)
    else
        error("wrong cellstate")
    end
    return cv

end 

function build_cohesive_cellvalue!(igashell, cellid::Int)
    cv_top = intdata(igashell).cell_values_cohesive_top
    cv_bot = intdata(igashell).cell_values_cohesive_bot
    cellstate = getcellstate(igashell, cellid)

    oop_cohesive_top_values, oop_cohesive_bot_values = _build_oop_cohesive_basisvalue!(igashell, cellstate)

    set_oop_basefunctions!(cv_top, oop_cohesive_top_values)
    set_oop_basefunctions!(cv_bot, oop_cohesive_bot_values)
    
    return cv_top, cv_bot
end

function build_facevalue!(igashell, faceidx::FaceIndex)
    cellid,faceid = faceidx
    cellstate = getcellstate(igashell, cellid)
    
    cv = intdata(igashell).cell_values_face
    oop_values = _build_oop_basisvalue!(igashell, cellstate, faceid)
 
    set_quadraturerule!(cv, get_face_qr(intdata(igashell), faceid))
    set_oop_basefunctions!(cv, oop_values)

    return cv
end 

function build_facevalue!(igashell::IGAShell{1,2}, edgeidx::VertexIndex)
    cellid, vertexid = edgeidx
    cellstate = getcellstate(igashell, cellid)

    vertexid = vertex_converter(igashell, vertexid)

    cv = intdata(igashell).cell_values_side
    
    oop_values = _build_oop_basisvalue!(igashell, cellstate)
    basisvalues_inplane = cached_side_basisvalues(intdata(igashell), vertexid)

    set_inp_basefunctions!(cv, basisvalues_inplane)
    set_oop_basefunctions!(cv, oop_values)

    return cv
end 

function build_facevalue!(igashell, edgeidx::EdgeIndex)
    cellid, edgeid = edgeidx
    cellstate = getcellstate(igashell, cellid)

    cv = intdata(igashell).cell_values_side
    
    oop_values = _build_oop_basisvalue!(igashell, cellstate)
    basisvalues_inplane = cached_side_basisvalues(intdata(igashell), edgeid)

    set_inp_basefunctions!(cv, basisvalues_inplane)
    set_oop_basefunctions!(cv, oop_values)

    return cv
end 

function build_facevalue!(igashell, edgeidx::EdgeInterfaceIndex)
    cellid, edgeid, face = edgeidx
    cellstate = getcellstate(igashell, cellid)

    cv = intdata(igashell).cell_values_interface
    
    oop_values = _build_oop_basisvalue!(igashell, cellstate, face)
    basisvalues_inplane = cached_side_basisvalues(intdata(igashell), edgeid)
    
    set_quadraturerule!(cv, get_face_qr(intdata(igashell), face))
    set_inp_basefunctions!(cv, basisvalues_inplane)
    set_oop_basefunctions!(cv, oop_values) 

    return cv
end 

function build_facevalue!(igashell, vertex::VertexInterfaceIndex)
    cellid, vertexid, face = vertex
    cellstate = getcellstate(igashell, cellid)
    
    vertexid = vertex_converter(igashell, vertexid)

    cv = intdata(igashell).cell_values_vertices
    
    oop_values = _build_oop_basisvalue!(igashell, cellstate, face)
    basisvalues_inplane = cached_vertex_basisvalues(intdata(igashell), vertexid)
    
    set_quadraturerule!(cv, get_face_qr(intdata(igashell), face))
    set_inp_basefunctions!(cv, basisvalues_inplane)
    set_oop_basefunctions!(cv, oop_values) 

    return cv
end 

function build_active_layer_dofs(igashell::IGAShell{dim_p, dim_s}, cellstate::CELLSTATE) where {dim_p,dim_s}

    if is_lumped(cellstate)
        return intdata(igashell).cache_values.active_layer_dofs_lumped
    elseif is_layered(cellstate)
        return intdata(igashell).cache_values.active_layer_dofs_layered
    elseif is_fully_discontiniuos(cellstate)
        return intdata(igashell).cache_values.active_layer_dofs_discont
    else
        return generate_active_layer_dofs(nlayers(igashell), ooplane_order(layerdata(igashell)), dim_s, JuAFEM.nnodes_per_cell(igashell), cellstate)
    end

end

function build_active_interface_dofs(igashell::IGAShell{dim_p, dim_s}, cellstate::CELLSTATE) where {dim_p,dim_s}

    if is_lumped(cellstate)
        return intdata(igashell).cache_values.active_interface_dofs_lumped
    elseif is_layered(cellstate)
        return intdata(igashell).cache_values.active_interface_dofs_layered
    elseif is_fully_discontiniuos(cellstate)
        return intdata(igashell).cache_values.active_interface_dofs_discont
    else
        return generate_active_interface_dofs(ninterfaces(igashell), ooplane_order(layerdata(igashell)), dim_s, JuAFEM.nnodes_per_cell(igashell), cellstate)
    end

end

function _build_oop_basisvalue!(igashell::IGAShell{dim_p,dim_s,T}, cellstate::CELLSTATE, faceidx::Int = -1) where {dim_p,dim_s,T}

    oop_values = BasisValues{1,T,1}[]

    for i in 1:JuAFEM.nnodes_per_cell(igashell)
        cp_state = get_cpstate(cellstate, i)

        if faceidx != -1
            cv_oop = cached_face_basisvalues(intdata(igashell), cp_state, faceidx)
        else
            cv_oop = cached_cell_basisvalues(intdata(igashell), cp_state)
        end

        push!(oop_values, cv_oop)
    end


    return oop_values
end

function _build_oop_cohesive_basisvalue!(igashell::IGAShell{dim_p,dim_s,T}, cellstate::CELLSTATE) where {dim_p,dim_s,T}
    
    oop_cohesive_top_values = BasisValues{1,T,1}[]
    oop_cohesive_bot_values = BasisValues{1,T,1}[]

    for i = 1:JuAFEM.nnodes_per_cell(igashell)
        cp_state = get_cpstate(cellstate, i)
        
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
    X = zeros(Vec{dim_s,T}, nnodes)
    Xᵇ = similar(X)
    celldofs = zeros(Int, nnodes)

    nqp_oop_per_layer = getnquadpoints_ooplane_per_layer(igashell)

    Δt = state.Δt

    V = 0
    @timeit "Shell loop" for (ic, cellid) in enumerate(igashell.cellset)
        cellstate = getcellstate(adapdata(igashell), ic)
        cv = build_cellvalue!(igashell, cellstate)
        active_layer_dofs = build_active_layer_dofs(igashell, cellstate)
        
        Ce = get_extraction_operator(intdata(igashell), ic)
        IGA.set_bezier_operator!(cv, Ce)

        ndofs = JuAFEM.ndofs_per_cell(dh, cellid)
        resize!(celldofs, ndofs)

        JuAFEM.cellcoords!(X, dh, cellid)
        JuAFEM.celldofs!(celldofs, dh, cellid)
        
        ue = state.d[celldofs]

        Xᵇ .= IGA.compute_bezier_points(Ce, X)
        @timeit "reinit1" reinit!(cv, Xᵇ)

        ⁿmaterialstates = state.prev_partstates[ic].materialstates
        materialstates = state.partstates[ic].materialstates

        for ilay in 1:nlayers(igashell)
            active_dofs = active_layer_dofs[ilay]

            ue_layer = ue[active_dofs]
            ndofs_layer = length(active_dofs)

            fe = zeros(T, ndofs_layer)
            ke = zeros(T, ndofs_layer, ndofs_layer)

            ⁿstates =  @view ⁿmaterialstates[:, ilay]
            states = @view materialstates[:, ilay]

            if assemtype == IGASHELL_STIFFMAT
                @timeit "integrate shell" _get_layer_forcevector_and_stiffnessmatrix!(
                                                    cv, 
                                                    ke, fe, 
                                                    getmaterial(layerdata(igashell)), states, 
                                                    ue_layer, ilay, nlayers(igashell), active_dofs, 
                                                    is_small_deformation_theory(layerdata(igashell)), IGASHELL_STIFFMAT, getwidth(layerdata(igashell)))
                
                assemble!(assembler, celldofs[active_dofs], ke, fe)
            elseif assemtype == IGASHELL_FSTAR

                @timeit "integrate shell" _get_layer_forcevector_and_stiffnessmatrix!(
                                        cv, 
                                        ke, fe, 
                                        getmaterial(layerdata(igashell)), states, 
                                        ue_layer, ilay, nlayers(igashell), active_dofs, 
                                        is_small_deformation_theory(layerdata(igashell)), IGASHELL_FSTAR, getwidth(layerdata(igashell)))

                state.system_arrays.fᴬ[celldofs[active_dofs]] += fe

            elseif assemtype == IGASHELL_DISSIPATION
                
            else
                error("Wrong option")
            end
            #=F!(ife, ike, u, ⁿms, ms) = _get_layer_forcevector_and_stiffnessmatrix!(
                cv, 
                ike, ife, 
                material(igashell), ms, ⁿms, 
                u, ilay, nlayers(igashell), active_dofs, 
                is_small_deformation_theory(layerdata(igashell)), getwidth(layerdata(igashell)))
                
            fe , ke = numdiff(F!, ue_layer, ⁿstates, states)=#

        end
    end
    
    
    icoords = zeros(Vec{dim_s,T}, nnodes*2)    
    A = 0.0
    @timeit "Interface loop" for (ic, cellid) in enumerate(igashell.cellset)

        cellstate = getcellstate(adapdata(igashell), ic)

        if is_lumped(cellstate) || is_layered(cellstate)
            continue
        end
        
        cv_cohesive_top, 
        cv_cohesive_bot = build_cohesive_cellvalue!(igashell, ic) 

        active_interface_dofs = build_active_interface_dofs(igashell, cellstate)

        interfacestates = state.partstates[ic].interfacestates

        ndofs = JuAFEM.ndofs_per_cell(dh,cellid)
        resize!(celldofs, ndofs)

        JuAFEM.cellcoords!(X, dh, cellid)
        JuAFEM.celldofs!(celldofs, dh, cellid)
        
        Ce = get_extraction_operator(intdata(igashell), ic)

        Xᵇ .= IGA.compute_bezier_points(Ce, X)
        
        for iint in 1:ninterfaces(igashell)      

            active_dofs = 1:JuAFEM.ndofs_per_cell(dh,ic)# active_interface_dofs[iint]
            
            if !is_interface_active(cellstate, iint)
                continue
            end
            
            states = @view interfacestates[:, iint]

            Δue = state.Δd[celldofs]
            ue = state.d[celldofs]
            due = state.v[celldofs]
            
            IGA.set_bezier_operator!(cv_cohesive_top, Ce)
            IGA.set_bezier_operator!(cv_cohesive_bot, Ce)

            ue_interface = @view ue[active_dofs]
            Δue_interface = @view Δue[active_dofs]

            ife = zeros(T, length(active_dofs))
            ike = zeros(T, length(active_dofs), length(active_dofs))  
            
            @timeit "reinit1" reinit!(cv_cohesive_top, Xᵇ)
            @timeit "reinit1" reinit!(cv_cohesive_bot, Xᵇ)

            if assemtype == IGASHELL_STIFFMAT
            @timeit "integrate_cohesive" A += integrate_cohesive_forcevector_and_stiffnessmatrix!(
                                                        cv_cohesive_top, cv_cohesive_bot,
                                                        interface_material(igashell), viscocity_parameter(layerdata(igashell)), 
                                                        states,
                                                        ike, ife,                
                                                        ue_interface, 
                                                        Δue_interface, 
                                                        Δt,
                                                        iint, ninterfaces(igashell),
                                                        active_dofs, getwidth(layerdata(igashell))) 
                assemble!(assembler, celldofs[active_dofs], ike, ife)
            elseif assemtype == IGASHELL_FSTAR
                ⁿinterfacestates = state.prev_partstates[ic].interfacestates
                ⁿstates =  @view ⁿinterfacestates[:, iint]

                @timeit "integrate_cohesive_fstar" integrate_cohesive_fstar!(
                                                            cv_cohesive_top, cv_cohesive_bot,
                                                            interface_material(igashell), viscocity_parameter(layerdata(igashell)), 
                                                            ⁿstates,
                                                            ike, ife,                
                                                            ue_interface, 
                                                            Δue_interface, 
                                                            Δt,
                                                            iint, ninterfaces(igashell),
                                                            active_dofs, getwidth(layerdata(igashell))) 
                    state.system_arrays.fᴬ[celldofs[active_dofs]] += ife
            elseif assemtype == IGASHELL_DISSIPATION
                ge = Base.RefValue(zero(T))
                @timeit "integrate_cohesive_dissi" integrate_dissipation!(
                                                            cv_cohesive_top, cv_cohesive_bot,
                                                            interface_material(igashell), viscocity_parameter(layerdata(igashell)), 
                                                            states,
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

            #=F!(ife, ike, u, ⁿms, ms) = integrate_cohesive_forcevector_and_stiffnessmatrix!(
                cv_cohesive_top, cv_cohesive_bot,
                interface_material(igashell), 
                ⁿms, ms,
                copy(ike), ife,
                u, 
                iint, ninterfaces(igashell),
                active_dofs, getwidth(layerdata(igashell))) 

                
            ife , ike = numdiff(F!, ue_interface, ⁿmaterialstates, materialstates)=#
            
            #=if !isapprox(norm(ike), norm(ike2), atol = 1e-0)
            end=#
            
        end
    end  

end

function assemble_massmatrix!( dh::JuAFEM.AbstractDofHandler, igashell::IGAShell{dim_p,dim_s,T}, system_arrays::SystemArrays) where {dim_p,dim_s,T}


end

function Five.post_part!(dh, igashell::IGAShell{dim_p,dim_s,T}, states) where {dim_s, dim_p, T}
    #if dim_s == 2
    #    return
    #end

    for (ic,cellid) in enumerate(igashell.cellset)#enumerate(CellIterator(dh, igashell.cellset))
        
        if ic != 223
            continue
        end

        cellstate = getcellstate(adapdata(igashell), ic)

        if !is_lumped(cellstate) && !is_layered(cellstate)
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
        cv = build_cellvalue!(igashell, cellstate)
        IGA.set_bezier_operator!(cv, Ce)
        reinit!(cv, Xᵇ)

        #Build basis_values for stress_recovory
        cv_sr = intdata(igashell).cell_values_sr
        oop_values = _build_oop_basisvalue!(igashell, cellstate)
        set_oop_basefunctions!(cv_sr, oop_values)
        IGA.set_bezier_operator!(cv_sr, Ce)
        reinit!(cv_sr, Xᵇ)

        recover_cell_stresses(srdata(igashell), σ_states, celldata, cv_sr, cv)
    end

end

function _get_layer_forcevector_and_stiffnessmatrix!(
                                cv::IGAShellValues{dim_s,dim_p,T}, 
                                ke::AbstractMatrix, fe::AbstractVector,
                                material, materialstate, 
                                ue_layer::AbstractVector{T}, ilay::Int, nlayers::Int, active_dofs::Vector{Int}, 
                                is_small_deformation_theory::Bool, calculate_what::IGASHELL_ASSEMBLETYPE, width::T) where {dim_s,dim_p,T}
                                
    ndofs_layer = length(active_dofs)

    nquadpoints_per_layer = getnquadpoints(cv) ÷ nlayers
    nquadpoints_ooplane_per_layer = getnquadpoints_ooplane(cv) ÷ nlayers

    δF = zeros(Tensor{2,dim_s,T,dim_s^2}, ndofs_layer)
    δɛ = zeros(Tensor{2,dim_s,T,dim_s^2}, ndofs_layer)
    g = zeros(Vec{dim_s,T}, dim_s)

    qp = (ilay-1) * nquadpoints_per_layer #Counter for the cell qp
    qpᴸ = 0 #Counter for the layer qp

    for _ in 1:nquadpoints_ooplane_per_layer
        for iqp in 1:getnquadpoints_inplane(cv)
            qp  +=1
            qpᴸ +=1 
            #R = calculate_R(g...)
            R = cv.R[iqp]
            #Covarient triad on deformed configuration
            for d in 1:dim_s
                g[d] = cv.G[qp][d] + function_parent_derivative(cv, qp, ue_layer, d, active_dofs)
            end

            
            # Deformation gradient
            F = zero(Tensor{2,dim_s,T})
            for i in 1:dim_s
                F += g[i]⊗cv.Gᴵ[qp][i]
            end

            for i in 1:ndofs_layer
                _δF = zero(Tensor{2,dim_s,T})
                for d in 1:dim_s
                    #Extract the d:th derivative wrt to parent coords \xi, \eta, \zeta
                    δg = basis_parent_derivative(cv, qp, active_dofs[i], d)
                    _δF += δg ⊗ cv.Gᴵ[qp][d]
                end
                δF[i] = _δF
            end
            
            if calculate_what === IGASHELL_STIFFMAT
                if is_small_deformation_theory
                    _calculate_linear_forces!(fe, ke, cv, 
                                                ilay, qpᴸ, qp, width,
                                                F, R, δF, δɛ,
                                                material, materialstate, 
                                                ndofs_layer)
                else
                    _calculate_nonlinear_forces!(fe, ke, cv, 
                                                ilay, qpᴸ, qp, width,
                                                F, R, δF, δɛ,
                                                material, materialstate, 
                                                ndofs_layer)
                end
            elseif calculate_what === IGASHELL_FSTAR
                _calculate_fstar!(fe, ke, cv, 
                                            ilay, qpᴸ, qp, width,
                                            F, R, δF, δɛ,
                                            material, materialstate, 
                                            ndofs_layer)
            else
                error("Nothing to caluclate")
            end
        end
    end 

end

function _calculate_linear_forces!(fe, ke, cv, ilay, layer_qp, qp, width, F::Tensor{2,dim_s}, R, δF, δɛ, material, materialstates, ndofs_layer) where {dim_s}
    ɛ = symmetric(F) - one(SymmetricTensor{2,dim_s})
    
    δɛ .= symmetric.(δF)
    dΩ = getdetJdV(cv,qp) * width

    #Rotate strain
     _̂ε = symmetric(R' ⋅ ɛ ⋅ R)
     _̂σ, ∂̂σ∂ɛ, new_matstate = Five.constitutive_driver(material[ilay], _̂ε, materialstates[layer_qp])
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
            
            ke[i,j] += (ɛC ⊡ δɛⱼ) * dΩ # can only assign to parent of the Symmetric wrapper
            
        end

    end

end

function _calculate_nonlinear_forces!(fe, ke, cv, ilay, layer_qp, qp, width, F, R, δF, δE, material, materialstates, ndofs_layer)
    dΩ = getdetJdV(cv,qp)*width

    E = symmetric(1/2 * (F' ⋅ F - one(F)))
    #Ê = symmetric(R' ⋅ E ⋅ R)

    S, ∂S∂E, new_matstate = Five.constitutive_driver(material[ilay], E, materialstates[layer_qp])
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

function _calculate_fstar!(fe, ke, cv, ilay, layer_qp, qp, width, F, R, δF, δE, material, materialstates, ndofs_layer)
    dΩ = getdetJdV(cv,qp)*width

    E = symmetric(1/2 * (F' ⋅ F - one(F)))

    S, ∂S∂E, new_matstate = Five.constitutive_driver(material[ilay], E, materialstates[layer_qp])

    # Hoist computations of δE
    for i in 1:ndofs_layer
        δFi = δF[i]
        δE[i] = symmetric(1/2*(δFi'⋅F + F'⋅δFi))
    end

    for i in 1:ndofs_layer
        δFi = δF[i]
        fe[i] += (δE[i] ⊡ S) * dΩ
    end

end

function integrate_cohesive_forcevector_and_stiffnessmatrix!(
    cv_top::IGAShellValues{dim_s,dim_p,T}, cv_bot,
    material::Five.AbstractMaterial, ξ::T,
    materialstate::AbstractArray{<:Five.AbstractMaterialState}, 
    ke::AbstractMatrix, 
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
        A += dΓ
        
        u₊ = zero(Vec{dim_s,T}); u₋ = zero(Vec{dim_s,T}) 
        #Δu₊ = zero(Vec{dim_s,T}); Δu₋ = zero(Vec{dim_s,T}) 
        for (i,j) in enumerate(active_dofs)
            u₊ += cv_top.N[j,qp] * ue[i]
            u₋ += cv_bot.N[j,qp] * ue[i]

           # Δu₊ += cv_top.U[j,qp] * Δue[i]
            #Δu₋ += cv_bot.U[j,qp] * Δue[i]
        end

        #ΔJ = Δu₊ - Δu₋
        J = u₊ - u₋
        Ĵ = R'⋅J
        
        #constitutive_driver
        t̂, ∂t∂Ĵ, new_matstate = Five.constitutive_driver(material, Ĵ, materialstate[qp-qp_offset])
        materialstate[qp-qp_offset] = new_matstate

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
            δui = basis_value(cv_top, qp, active_dofs[i]) - basis_value(cv_bot, qp, active_dofs[i])

            fe[i] += (t ⋅ δui) * dΓ
            for j in 1:ndofs#length(active_dofs)
                δuj = basis_value(cv_top, qp, active_dofs[j]) - basis_value(cv_bot, qp, active_dofs[j])

                ke[i,j] += δui⋅∂t∂J⋅δuj * dΓ
            end
        end
    end
    #println("")
    return A
end

function integrate_cohesive_fstar!(
    cv_top::IGAShellValues{dim_s,dim_p,T}, cv_bot,
    material::Five.AbstractMaterial, ξ::T,
    materialstate::AbstractArray{<:Five.AbstractMaterialState}, 
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
        for (i,j) in enumerate(active_dofs)
            u₊ += cv_top.N[j,qp] * ue[i]
            u₋ += cv_bot.N[j,qp] * ue[i]
        end

        J = u₊ - u₋
        Ĵ = R'⋅J
        
        #constitutive_driver
        t̂, ∂t∂Ĵ, new_matstate = Five.constitutive_driver(material, Ĵ, materialstate[qp-qp_offset])

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
        g, dgdĴ = Five.constitutive_driver_dissipation(material, Ĵ, materialstate[qp-qp_offset])
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
                                     ue::AbstractVector{T},
                                     active_dofs::AbstractVector{Int}) where {dim_p,dim_s,T}


    g₊ = zeros(Vec{dim_s,T},dim_p)
    g₋ = zeros(Vec{dim_s,T},dim_p)
    for d in 1:dim_p
        g₊[d] = cv_top.G[qp][d] + function_parent_derivative(cv_top, qp, ue, d, active_dofs)
        g₋[d] = cv_bot.G[qp][d] + function_parent_derivative(cv_bot, qp, ue, d, active_dofs)
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
    #@assert( det(R)≈1.0 )

    dV = detJ * cv_top.qr.weights[qp]
    
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
