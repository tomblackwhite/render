#include <scene.hh>
namespace App {
using glm::vec3;
using std::holds_alternative;
using std::index_sequence;
using std::initializer_list;
using std::vector;

SceneManager::SceneManager(VulkanMemory *memory, PipelineFactory *factory,
                           const std::string &homePath)
    : m_vulkanMemory(memory), m_pipelineFactory(factory), m_homePath(homePath),
      m_assetManager(AssetManager::instance(homePath)),
      m_scene(memory, factory, homePath) {}

void SceneManager::init() {

  loadScene(m_mainScene);
  // 绑定脚本
  bindScript();

  auto &map = m_nodeContainer.map;
  // 初始化脚本
  ranges::for_each(map, [](auto const &keyValue) {
    auto *node = keyValue.second.get();

    auto *pscript = node->script();
    if (pscript) {
      pscript->init();
    }
  });

  // 执行脚本后根据node状态准备showScene
  showScene(m_mainScene);
}

void SceneManager::update() {

  ranges::for_each(m_nodeContainer.map, [](auto const &keyValue) {
    auto *node = keyValue.second.get();
    auto *pscript = node->script();
    if (pscript) {
      pscript->update();
    }
  });
}

void SceneManager::physicalUpdate() {

  auto &map = m_nodeContainer.map;
  ranges::for_each(map, [](auto const &keyValue) {
    auto *node = keyValue.second.get();
    auto *pscript = node->script();
    if (pscript) {
      pscript->physicalUpdate();
    }
  });
}

void SceneManager::loadScene(const string &scene) {
  m_factory->createScene(m_assetManager, scene, &m_nodeFactory,
                         &m_nodeContainer);
}

void SceneManager::bindScript() {
  auto *rootNode = m_nodeContainer.map[m_mainScene].get();

  if (rootNode != nullptr) {
    auto rootScript = std::make_unique<RootScript>(rootNode, &m_nodeFactory);
    rootNode->setScript(std::unique_ptr<Script>(rootScript.release()));
  }
}

void SceneManager::showScene(const string &scene) {

  auto visitor = [&showMap = this->m_scene.showMap,
                  &vulkanMemory = this->m_vulkanMemory, &scene = this->m_scene,
                  this](Node *pNode) {
    // 是否显示
    if ((pNode != nullptr) && pNode->visible()) {
      if (auto *mesh = dynamic_cast<MeshInstance *>(pNode); mesh) {

        mesh->modelMatrix = this->getTransform(mesh);
        showMap.insert_or_assign(mesh->name, mesh);
      }
      if (auto *camera = dynamic_cast<Camera *>(pNode); camera) {
        scene.camera.proj = camera->projection;
        glm::vec3 scale;
        glm::quat rotate;
        glm::vec3 translation;
        glm::vec3 skew;
        glm::vec4 per;
        auto transform = this->getTransform(camera);
        glm::decompose(transform, scale, rotate, translation, skew, per);

        //
        auto view = glm::transpose(glm::toMat4(rotate)) *
                    glm::translate(glm::mat4(1), -translation);
        ;

        scene.camera.view = view;
        camera->view = view;
        scene.camera.viewProj = scene.camera.proj * scene.camera.view;
      }
    }
  };

  visitNode(m_mainScene, visitor);

  m_scene.meshShowMap.clear();

  for (auto &[key, meshInstance] : m_scene.showMap) {
    if (auto iter = m_scene.meshShowMap.find(meshInstance->mesh);
        iter != m_scene.meshShowMap.end()) {
      iter->second.push_back(meshInstance->modelMatrix);
    } else {
      m_scene.meshShowMap.insert(
          {meshInstance->mesh, {meshInstance->modelMatrix}});
    }
  }

  auto meshsView =
      views::all(m_scene.meshShowMap) |
      views::transform([](auto &elem) -> Mesh * { return elem.first; });
  std::vector<Mesh *> meshes(meshsView.begin(), meshsView.end());

#ifdef DEBUG
  // debug info
  auto meshCount = meshsView.size();
#endif

  // 上传到gpu
  std::unordered_set<Image *> images{};
  std::unordered_set<Texture *> textures{};
  std::unordered_set<Material *> materials{};

  auto checkTexture = [&textures, &images](Material::TextureIterator iterator) {
    if (iterator != Material::TextureIterator()) {
      if (!*(iterator->imageView)) {
        textures.insert(&*iterator);

        if (iterator->imageIterator != Texture::ImageIterator()) {
          if ((iterator->imageIterator->image.get()) == nullptr) {
            images.insert(&*iterator->imageIterator);
          }
        }
      }
    }
  };
  for (auto *mesh : meshes) {

    for (auto &subMesh : mesh->subMeshs) {
      if (subMesh.material != nullptr) {
        auto *material = subMesh.material;
        if (!*(material->textureSet)) {
          materials.insert(material);
          auto &pbr = material->pbr;
          checkTexture(pbr.baseColorTexture);
          checkTexture(pbr.metallicRoughnessTexture);
        }
      }
    }
  }

  m_scene.vertexBuffer = m_vulkanMemory->uploadMeshesByTransfer(
      meshsView, images, m_scene.meshShowMap);

  vk::ImageSubresourceRange imageViewRange{};
  imageViewRange.setAspectMask(vk::ImageAspectFlagBits::eColor);
  imageViewRange.setBaseMipLevel(0);
  imageViewRange.setLevelCount(1);
  imageViewRange.setBaseArrayLayer(0);
  imageViewRange.setLayerCount(1);

  vk::ImageViewCreateInfo imageViewInfo{};
  imageViewInfo.setViewType(vk::ImageViewType::e2D);

  imageViewInfo.setSubresourceRange(imageViewRange);

  vk::SamplerCreateInfo samplerInfo{};
  samplerInfo.setAddressModeU(vk::SamplerAddressMode::eRepeat);
  samplerInfo.setAddressModeV(vk::SamplerAddressMode::eRepeat);
  samplerInfo.setAddressModeW(vk::SamplerAddressMode::eRepeat);
  samplerInfo.setAnisotropyEnable(VK_TRUE);
  samplerInfo.setMaxAnisotropy(16);

  for (auto *texture : textures) {
    imageViewInfo.setImage(texture->imageIterator->image.get());
    imageViewInfo.setFormat(texture->format);
    texture->imageView =
        m_vulkanMemory->m_pDevice->createImageView(imageViewInfo);
    samplerInfo.setMagFilter(texture->magFilter);
    samplerInfo.setMinFilter(texture->minFilter);

    TextureFilter filter{.magFilter = texture->magFilter,
                         .minFilter = texture->minFilter};

    vk::Sampler sampler{nullptr};

    if (auto iter = m_scene.samplerMap.find(filter);
        iter != m_scene.samplerMap.end()) {
      sampler = *(iter->second);
    } else {

      raii::Sampler createSampler =
          m_vulkanMemory->m_pDevice->createSampler(samplerInfo);
      sampler = *createSampler;
      m_scene.samplerMap.insert_or_assign(filter, std::move(createSampler));
    }

    texture->sampler = sampler;
  }

  // auto first = textures.begin();

  // imageViewInfo.setImage((*first)->imageIterator->image.get());
  // imageViewInfo.setFormat((*first)->format);
  // raii::ImageView testView =
  //     m_vulkanMemory->m_pDevice->createImageView(imageViewInfo);

  // vk::DescriptorImageInfo imageDescriptorInfo{
  //     .sampler = (*first)->sampler,
  //     .imageView = (*(*first)->imageView),
  //     .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};

  // 设置大小避免迭代器失效
  std::vector<std::vector<vk::DescriptorImageInfo>> descriptorImages{};
  descriptorImages.reserve(materials.size());
  std::vector<vk::WriteDescriptorSet> writeSets{};
  for (auto *material : materials) {
    auto textureSets =
        m_vulkanMemory->createDescriptorSet(*(m_scene.textureSetLayout));
    material->textureSet = std::move(textureSets[0]);
    vk::DescriptorImageInfo baseColorDescriptorInfo{};
    baseColorDescriptorInfo.setImageLayout(
        vk::ImageLayout::eShaderReadOnlyOptimal);
    baseColorDescriptorInfo.setSampler(material->pbr.baseColorTexture->sampler);
    baseColorDescriptorInfo.setImageView(
        *(material->pbr.baseColorTexture->imageView));

    vk::DescriptorImageInfo metallicRoughnessDescriptorInfo{};
    metallicRoughnessDescriptorInfo.setImageLayout(
        vk::ImageLayout::eShaderReadOnlyOptimal);

    if (material->pbr.metallicRoughnessTexture != Material::TextureIterator()) {
      metallicRoughnessDescriptorInfo.setSampler(
          material->pbr.metallicRoughnessTexture->sampler);
      metallicRoughnessDescriptorInfo.setImageView(
          *(material->pbr.metallicRoughnessTexture->imageView));
    }else{
      metallicRoughnessDescriptorInfo.setSampler(material->pbr.baseColorTexture->sampler);
    }

    descriptorImages.push_back(
        {baseColorDescriptorInfo, metallicRoughnessDescriptorInfo});
    vk::WriteDescriptorSet writeSet{};
    writeSet.setDstSet(*(material->textureSet));
    writeSet.setDstBinding(0);
    writeSet.setDescriptorCount(descriptorImages.back().size());
    writeSet.setImageInfo(descriptorImages.back());
    writeSet.setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
    writeSets.push_back(writeSet);
  }

  vk::DescriptorBufferInfo descriptorBufferInfo{};
  descriptorBufferInfo.setBuffer(m_scene.vertexBuffer.objectBuffer.get());
  descriptorBufferInfo.setOffset(0);
  descriptorBufferInfo.setRange(
      m_scene.vertexBuffer.objectBuffer.get_deleter().m_size);

  // 绑定descriptorSet
  vk::WriteDescriptorSet writeSet{};
  writeSet.setDstSet(*(m_scene.objectSet));
  writeSet.setDstBinding(0);
  writeSet.setDescriptorType(vk::DescriptorType::eStorageBufferDynamic);
  writeSet.setBufferInfo(descriptorBufferInfo);


  writeSets.push_back(writeSet);
  m_vulkanMemory->m_pDevice->updateDescriptorSets(
      writeSets, std::initializer_list<vk::CopyDescriptorSet>{});

  // 上传camera
  m_vulkanMemory->uploadByTransfer(
      m_scene.cameraBuffer, views::single(std::span(&m_scene.camera, 1)));
}

glm::mat4 SceneManager::getTransform(Node *node) {

  auto &map = m_nodeContainer.map;

  glm::mat4 trans{node->transform()};

  auto parentKey = node->parent();
  for (; !parentKey.empty();) {
    if (auto iter = map.find(parentKey); iter != map.end()) {
      trans = iter->second->transform() * trans;
      parentKey = iter->second->parent();
    } else {
      //
      spdlog::warn("some node lost so set transform to default");
      return glm::mat4{1};
    }
  }
  return trans;
}

void SceneManager::visitNode(const string &key,
                             std::function<void(Node *)> const &visitor) {
  auto &map = m_nodeContainer.map;
  auto &children = map[key]->children();

  for (auto &elemKey : children) {
    auto &elem = map[elemKey];
    visitor(elem.get());
    visitNode(elemKey, visitor);
  }
}

void SceneFactory::createScene(AssetManager &assetManager,
                               const std::string &sceneKey,
                               NodeFactory *nodeFactory,
                               NodeContainer *container) {

  auto &map = container->map;
  auto &tree = container->tree;
  auto &meshMap = container->meshMap;
  auto &imageMap = container->imageMap;
  auto &textureMap = container->textureMap;
  auto &materialMap = container->materailMap;

  auto &model = assetManager.getScene(sceneKey);
  auto &scene = model.scenes[model.defaultScene];

  auto images = std::make_unique<std::vector<Image>>();

  auto *pImages = images.get();
  for (auto &image : model.images) {
    images->push_back(createImage(image));
  }
  imageMap.insert_or_assign(sceneKey, std::move(images));

  auto textures = std::make_unique<std::vector<Texture>>();
  auto *pTexture = textures.get();
  for (auto &texture : model.textures) {
    textures->push_back(createTexture(texture, pImages, model));
  }
  textureMap.insert_or_assign(sceneKey, std::move(textures));

  auto materials = std::make_unique<std::vector<Material>>();
  auto *pMaterial = materials.get();
  for (auto &material : model.materials) {
    materials->push_back(createMaterial(material, pTexture));
  }
  materialMap.insert_or_assign(sceneKey, std::move(materials));

  auto meshes = std::make_unique<std::vector<Mesh>>();
  auto *pMeshes = meshes.get();
  for (auto &mesh : model.meshes) {
    meshes->push_back(createMesh(mesh, model, pMaterial));
  }
  meshMap.insert_or_assign(sceneKey, std::move(meshes));

  for (auto &node : model.nodes) {
    createNode(node, model, pMeshes, nodeFactory);
  }
  std::vector<std::string> sceneNodes;
  // 递归构建Node Tree
  for (auto index : scene.nodes) {
    auto &node = model.nodes[index];
    sceneNodes.push_back(node.name);
    createNodeTree(node, model, map, tree);
  }

  // 创建rootNode
  auto *rootNode = nodeFactory->createNode<Node>(sceneKey);
  rootNode->childern() = std::move(sceneNodes);
  // sceneKey as root of scene nodes
}

void SceneFactory::createNode(tinygltf::Node const &node,
                              tinygltf::Model &model, std::vector<Mesh> *meshes,
                              NodeFactory *nodeFactory) {

  Node *result = nullptr;

  tinygltf::Image image;
  // 创建mesh
  if (node.mesh != -1) {

    auto *meshIns = nodeFactory->createNode<MeshInstance>(node.name);
    meshIns->mesh = &*(meshes->begin() + node.mesh);
  } else if (node.camera != -1) {
    auto *camera =
        createCamera(model.cameras[node.camera], node.name, nodeFactory);
    result = std::move(camera);

  } else {
    result = nodeFactory->createNode<Node>(node.name);
  }

  if (!node.scale.empty()) {
    auto scale =
        castToGLMType<glm::vec3>(node.scale, index_sequence<0, 1, 2>{});
    result->setScale(scale);
  }
  if (!node.rotation.empty()) {

    auto quat =
        castToGLMType<glm::quat>(node.rotation, index_sequence<3, 0, 1, 2>{});
    result->setRotation(quat);
  }

  if (!node.translation.empty()) {

    auto translation =
        castToGLMType<glm::vec3>(node.translation, index_sequence<0, 1, 2>{});
    result->setTranslation(translation);
  }
  if (!node.matrix.empty()) {
    auto matrix =
        castToGLMType<glm::mat4>(node.matrix, std::make_index_sequence<16>{});
    result->setTransform(matrix);
  }
}

void SceneFactory::createNodeTree(const tinygltf::Node &node,
                                  tinygltf::Model &model, NodeMap &map,
                                  NodeTree &tree) {
  if (node.children.empty()) {
    return;
  }
  // NodeTree::value_type keyValue{node.name, NodeTree::mapped_type{}};
  // tree.insert(keyValue);

  std::vector<std::string> sceneNodes;
  // 构建场景路径。
  auto &currentNode = map[node.name];

  for (auto index : node.children) {

    // 设置node 中路径信息。
    auto const &nodeKey = model.nodes[index].name;
    auto &childrenNode = map[nodeKey];
    childrenNode->parent() = node.name;

    currentNode->childern().push_back(nodeKey);
    sceneNodes.push_back(model.nodes[index].name);
    createNodeTree(model.nodes[index], model, map, tree);
  }
}

// Mesh SceneFactory::createMesh(int meshIndex, tinygltf::Model &model,
//                               std::vector<tinygltf::Buffer> &buffers) {

//   Mesh result;
//   auto &mesh = model.meshes[meshIndex];

//   // 构建mesh
//   auto vec3Attributes = std::to_array<std::string>({"POSITION", "NORMAL"});
//   for (auto const &primitive : mesh.primitives) {

//     // 不存在indice 不会显示。

//     if (primitive.indices != -1) {

//       Mesh::SubMesh subMesh;
//       auto const &accessor = model.accessors[primitive.indices];
//       auto indexBuffer = createSpanBuffer(accessor, model, buffers);

//       if (std::holds_alternative<Mesh::IndexSpanType>(indexBuffer)) {
//         subMesh.indices = std::get<Mesh::IndexSpanType>(indexBuffer);

//         std::map<int, std::span<glm::vec2>> texCoords;
//         // 构建submesh中的attributes
//         auto const &attributes = primitive.attributes;
//         for (auto const &attri : attributes) {

//           if (auto iter = ranges::find(vec3Attributes, attri.first);
//               iter != ranges::end(vec3Attributes)) {

//             auto const &accessor = model.accessors[attri.second];
//             // 构建属性span buffer
//             auto spanBuffer = createSpanBuffer(accessor, model, buffers);

//             if (*iter == "POSITION" &&
//                 holds_alternative<std::span<glm::vec3>>(spanBuffer)) {

//               subMesh.positions = std::get<std::span<glm::vec3>>(spanBuffer);
//             } else if (*iter == "NORMAL" &&
//                        holds_alternative<std::span<glm::vec3>>(spanBuffer)) {
//               subMesh.normals = std::get<std::span<glm::vec3>>(spanBuffer);
//             } else {
//               auto viewStrings = views::split(*iter, "_");
//               std::vector<std::string_view> viewStrs(viewStrings.begin(),
//                                                      viewStrings.end());
//               if (viewStrs.size() == 2) {
//                 std::string indexStr(viewStrs[1]);
//                 auto indexNumber = std::stoi(indexStr);

//                 if (viewStrs[0] == "TEXCOORD" &&
//                     holds_alternative<std::span<glm::vec2>>(spanBuffer)) {
//                   texCoords.insert_or_assign(
//                       indexNumber,
//                       std::get<std::span<glm::vec2>>(spanBuffer));
//                 }

//               } else {

//                 spdlog::warn("{}'s type in asset is not match  {}'s type' ",
//                              mesh.name, *iter);
//               }
//             }
//           }
//         }

//         // 设置texCoords
//         for (auto &texCoordElem : texCoords) {
//           subMesh.texCoords.push_back(texCoordElem.second);
//         }

//         if (primitive.material != -1) {

//           auto &currentMat = model.materials.at(primitive.material);
//           auto &pbrCurrent = currentMat.pbrMetallicRoughness;
//         }

//         result.subMeshs.push_back(subMesh);
//         // result.indexCount+=subMesh.indices.size();
//         // result.vertexCount+=subMesh.positions.size();
//       } else {
//         // index type 不匹配时
//         auto indexType = vk::IndexTypeValue<IndexType>::value;
//         spdlog::warn("{}'s index type {} in gltf asset is not match indices's
//         "
//                      "type {} ,so don't appear\n",
//                      mesh.name, accessor.componentType,
//                      std::to_underlying(indexType));
//       }
//     } else {
//       spdlog::warn(
//           "{} primitive don't have indices attribute, so don't appear \n",
//           mesh.name);
//     }
//   }
//   return result;
// }
// Texture SceneFactory::createTexture(const tinygltf::TextureInfo &info,
//                                     tinygltf::Model &model) {
//   auto getFilter = [](int filterNumber) {
//     switch (filterNumber) {
//     case 9728:
//       return vk::Filter::eNearest;
//     case 9729:
//       return vk::Filter::eLinear;
//     default:
//       return vk::Filter::eLinear;
//     }
//   };

//   auto &textureInfo = model.textures[info.index];

//   Texture texture;
//   texture.coordIndex = info.texCoord;

//   if (textureInfo.sampler != -1) {
//     auto &sampler = model.samplers[textureInfo.sampler];
//     texture.magFilter = getFilter(sampler.magFilter);
//     texture.minFilter = getFilter(sampler.minFilter);
//   }
//   if (textureInfo.source != -1) {
//     auto &image = model.images[textureInfo.source];

//     vk::Format imageFormat = vk::Format::eR8G8B8Sint;

//     if (image.bits == 8) {
//       if (image.component == 3) {
//         imageFormat = vk::Format::eR8G8B8Sint;
//       } else if (image.component == 4) {
//         imageFormat = vk::Format::eR8G8B8A8Sint;
//       }
//     } else if (image.bits == 16) {
//       if (image.component == 3) {
//         imageFormat = vk::Format::eR16G16B16Sint;
//       } else if (image.component == 4) {
//         imageFormat = vk::Format::eR16G16B16A16Sint;
//       }
//     }

//     texture.data =
//         std::span<unsigned char>{image.image.data(), image.image.size()};

//     return texture;
//   }
// }

Camera *SceneFactory::createCamera(const tinygltf::Camera &camera,
                                   const std::string &name,
                                   NodeFactory *nodeFactory) {
  auto *result = nodeFactory->createNode<Camera>(name);
  if (camera.type == "perspective") {
    auto const &perpective = camera.perspective;

    // finite
    if (perpective.zfar != 0.0) {

      result->projection =
          glm::perspective(perpective.yfov, perpective.aspectRatio,
                           perpective.znear, perpective.zfar);
    } else {
      result->projection = glm::infinitePerspective(
          perpective.yfov, perpective.aspectRatio, perpective.znear);
    }
  } else if (camera.type == "orthographic") {
    auto const &orth = camera.orthographic;
    result->projection = glm::ortho(0.0, 2 * orth.xmag, 0.0, 2 * orth.ymag,
                                    orth.znear, orth.zfar);
  } else {
    // do nothing
    spdlog::warn("camera don't have type");
  }
  return result;
}
GlTFSpanVariantType
SceneFactory::createSpanBuffer(const tinygltf::Accessor &acessor,
                               const tinygltf::Model &model,
                               std::vector<tinygltf::Buffer> &buffers) {

  auto const &bufferView = model.bufferViews[acessor.bufferView];
  auto &buffer = buffers[bufferView.buffer];

  auto *bufferStart =
      buffer.data.data() + bufferView.byteOffset + acessor.byteOffset;

  auto componentType = GLTFComponentType{acessor.componentType};
  auto type = GLTFType{acessor.type};

  GlTFSpanVariantType result;

  // 根据类型返回对应span
  if (componentType == GLTFComponentType::Signed8 && type == GLTFType::Scalar) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed8, GLTFType::Scalar>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed8 && type == GLTFType::Scalar) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed8, GLTFType::Scalar>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Unsigned8 &&
      type == GLTFType::Scalar) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Unsigned8, GLTFType::Scalar>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed16 &&
      type == GLTFType::Scalar) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed16, GLTFType::Scalar>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Unsigned16 &&
      type == GLTFType::Scalar) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Unsigned16, GLTFType::Scalar>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed32 &&
      type == GLTFType::Scalar) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed32, GLTFType::Scalar>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::SignedFloat32 &&
      type == GLTFType::Scalar) {
    using currentType =
        getGLTFType_t<GLTFComponentType::SignedFloat32, GLTFType::Scalar>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed8 && type == GLTFType::Vec2) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed8, GLTFType::Vec2>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Unsigned8 && type == GLTFType::Vec2) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Unsigned8, GLTFType::Vec2>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed16 && type == GLTFType::Vec2) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed16, GLTFType::Vec2>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Unsigned16 &&
      type == GLTFType::Vec2) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Unsigned16, GLTFType::Vec2>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed32 && type == GLTFType::Vec2) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed32, GLTFType::Vec2>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::SignedFloat32 &&
      type == GLTFType::Vec2) {
    using currentType =
        getGLTFType_t<GLTFComponentType::SignedFloat32, GLTFType::Vec2>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed8 && type == GLTFType::Vec3) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed8, GLTFType::Vec3>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Unsigned8 && type == GLTFType::Vec3) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Unsigned8, GLTFType::Vec3>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed16 && type == GLTFType::Vec3) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed16, GLTFType::Vec3>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Unsigned16 &&
      type == GLTFType::Vec3) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Unsigned16, GLTFType::Vec3>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed32 && type == GLTFType::Vec3) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed32, GLTFType::Vec3>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::SignedFloat32 &&
      type == GLTFType::Vec3) {
    using currentType =
        getGLTFType_t<GLTFComponentType::SignedFloat32, GLTFType::Vec3>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed8 && type == GLTFType::Vec4) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed8, GLTFType::Vec4>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Unsigned8 && type == GLTFType::Vec4) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Unsigned8, GLTFType::Vec4>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed16 && type == GLTFType::Vec4) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed16, GLTFType::Vec4>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Unsigned16 &&
      type == GLTFType::Vec4) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Unsigned16, GLTFType::Vec4>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed32 && type == GLTFType::Vec4) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed32, GLTFType::Vec4>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::SignedFloat32 &&
      type == GLTFType::Vec4) {
    using currentType =
        getGLTFType_t<GLTFComponentType::SignedFloat32, GLTFType::Vec4>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed8 && type == GLTFType::Mat2) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed8, GLTFType::Mat2>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Unsigned8 && type == GLTFType::Mat2) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Unsigned8, GLTFType::Mat2>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed16 && type == GLTFType::Mat2) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed16, GLTFType::Mat2>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Unsigned16 &&
      type == GLTFType::Mat2) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Unsigned16, GLTFType::Mat2>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed32 && type == GLTFType::Mat2) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed32, GLTFType::Mat2>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::SignedFloat32 &&
      type == GLTFType::Mat2) {
    using currentType =
        getGLTFType_t<GLTFComponentType::SignedFloat32, GLTFType::Mat2>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed8 && type == GLTFType::Mat3) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed8, GLTFType::Mat3>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Unsigned8 && type == GLTFType::Mat3) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Unsigned8, GLTFType::Mat3>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed16 && type == GLTFType::Mat3) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed16, GLTFType::Mat3>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Unsigned16 &&
      type == GLTFType::Mat3) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Unsigned16, GLTFType::Mat3>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed32 && type == GLTFType::Mat3) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed32, GLTFType::Mat3>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::SignedFloat32 &&
      type == GLTFType::Mat3) {
    using currentType =
        getGLTFType_t<GLTFComponentType::SignedFloat32, GLTFType::Mat3>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed8 && type == GLTFType::Mat4) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed8, GLTFType::Mat4>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Unsigned8 && type == GLTFType::Mat4) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Unsigned8, GLTFType::Mat4>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed16 && type == GLTFType::Mat4) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed16, GLTFType::Mat4>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Unsigned16 &&
      type == GLTFType::Mat4) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Unsigned16, GLTFType::Mat4>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed32 && type == GLTFType::Mat4) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed32, GLTFType::Mat4>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::SignedFloat32 &&
      type == GLTFType::Mat4) {
    using currentType =
        getGLTFType_t<GLTFComponentType::SignedFloat32, GLTFType::Mat4>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  return result;
}

} // namespace App
