# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# DoubleMaterial - Special case where one has to lump two materials together
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

struct IGAShellMaterial{N,LM,IM} <: Five.AbstractMaterial
    layer_materials::LayeredMaterial{N,LM}
    interface_material::IM #Probobly only need one interface_material

    function IGAShellMaterial(lm::LayeredMaterial{N,LM}, im::IM) where {N,LM,IM}
        new{N,LM,IM}(lm,im)
    end

end

struct IGAShellMaterialState{LM,IM} <: Five.AbstractMaterialState
    layer_material_states::LM
    interface_material_states::IM
end

function IGAShellMaterialState(mat::IGAShellMaterial{N,LM,IM}, layer::Int) where {N,LM,IM}
    lm_state = LM(mat.layer_materials[layer])
    im_state = LM(mat.interface_material)
    return IGAShellMaterialState{LM,IM}(lm_state, im_state)
end

function get_material_state_type(m::IGAShellMaterial{N,LM,IM}) where {N,LM,IM}
    return IGAShellMaterialState{LM,IM}
end

function construct_igashell_materials(lm::AbstractMaterial, im::MatCohesive, nlayers::Int)

    layermats = [deepcopy(lm) for i in 1:nlayers]
    intermats = [deepcopy(im) for i in 1:nlayers-1]

    _lm = LayeredMaterial(NTuple{nlayers}(layermats))
    
    return IGAShellMaterial(_lm,im)

end