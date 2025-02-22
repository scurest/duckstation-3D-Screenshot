// SPDX-FileCopyrightText: 2019-2022 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#include <QtWidgets/QMessageBox>
#include <algorithm>

#include "foldersettingswidget.h"
#include "settingswindow.h"
#include "settingwidgetbinder.h"

FolderSettingsWidget::FolderSettingsWidget(SettingsWindow* dialog, QWidget* parent) : QWidget(parent)
{
  SettingsInterface* sif = dialog->getSettingsInterface();

  m_ui.setupUi(this);

  SettingWidgetBinder::BindWidgetToFolderSetting(sif, m_ui.cache, m_ui.cacheBrowse, m_ui.cacheOpen, m_ui.cacheReset,
                                                 "Folders", "Cache", Path::Combine(EmuFolders::DataRoot, "cache"));
  SettingWidgetBinder::BindWidgetToFolderSetting(sif, m_ui.covers, m_ui.coversBrowse, m_ui.coversOpen, m_ui.coversReset,
                                                 "Folders", "Covers", Path::Combine(EmuFolders::DataRoot, "covers"));
  SettingWidgetBinder::BindWidgetToFolderSetting(sif, m_ui.screenshots, m_ui.screenshotsBrowse, m_ui.screenshotsOpen,
                                                 m_ui.screenshotsReset, "Folders", "Screenshots",
                                                 Path::Combine(EmuFolders::DataRoot, "screenshots"));
  SettingWidgetBinder::BindWidgetToFolderSetting(sif, m_ui.screenshots3D, m_ui.screenshots3DBrowse, m_ui.screenshots3DOpen,
                                                 m_ui.screenshots3DReset, "Folders", "Screenshots3D",
                                                 Path::Combine(EmuFolders::DataRoot, "screenshots_3d"));
  SettingWidgetBinder::BindWidgetToFolderSetting(sif, m_ui.saveStates, m_ui.saveStatesBrowse, m_ui.saveStatesOpen,
                                                 m_ui.saveStatesReset, "Folders", "SaveStates",
                                                 Path::Combine(EmuFolders::DataRoot, "savestates"));
}

FolderSettingsWidget::~FolderSettingsWidget() = default;
