cmake_minimum_required(VERSION 2.8.3)

project(custom_stage_worlds)

find_package(catkin REQUIRED)


catkin_package()


install(DIRECTORY
  worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
