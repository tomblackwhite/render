#include "mainwindow.hh"
#include <QApplication>
#include <QLayout>
#include <QVulkanInstance>
#include <QWindow>
#include <iostream>
#include <memory>
#include <window.hh>

int main(int argc, char *argv[]) {
  auto resultcode = 0;
  try {
    QApplication app(argc, argv);

    std::string path = app.applicationDirPath().toStdString();

    auto qVulkanInstance = std::make_unique<QVulkanInstance>();

    // auto vulkanWindow= std::make_unique<VulkanWindow>();
    MainWindow w;
    auto vulkanGameWindow =
        std::make_unique<VulkanGameWindow>(qVulkanInstance.get(), path,&w);
    auto *widget = w.centralWidget();
    auto *layout = widget->layout();
    layout->addWidget(
        QWidget::createWindowContainer(vulkanGameWindow.release()));
    w.show();
    resultcode = app.exec();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return resultcode;
  }

  return EXIT_SUCCESS;
}
