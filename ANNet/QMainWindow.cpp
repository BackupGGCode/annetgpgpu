#include <iostream>
#include <QEdge.h>
#include <QNode.h>
#include <gui/QLayer.h>
#include <gui/QMainWindow.h>
#include <gui/utils/stylehelper.h>
#include <gui/utils/manhattanstyle.h>  //"manhattanstyle.h"
#include <math/ANFunctions.h>
#include <containers/ANTrainingSet.h>


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    using namespace Core;
    using namespace Core::Internal;

    QCoreApplication::setApplicationName(QLatin1String("ANNetDesigner"));
    QString baseName = QApplication::style()->objectName();
#ifdef Q_WS_X11
    if (baseName == QLatin1String("windows")) {
        // Sometimes we get the standard windows 95 style as a fallback
        // e.g. if we are running on a KDE4 desktop
        QByteArray desktopEnvironment = qgetenv("DESKTOP_SESSION");
        if (desktopEnvironment == "kde")
            baseName = QLatin1String("plastique");
        else
            baseName = QLatin1String("cleanlooks");
    }
#endif
    m_pANNet 		= NULL;
    m_pTrainingSet 	= NULL;

    qApp->setStyle(new ManhattanStyle(baseName));
    Utils::StyleHelper::setBaseColor(Qt::darkGray);

    m_ActionsBar    = new QToolBar;
    m_pTabBar       = new FancyTabWidget;
    m_pActionBar 	= new FancyActionBar;

    m_pViewer       = new Viewer;
    m_pCustomPlot   = new QCustomPlot;
    m_pInputDial    = new IOForm;
    m_pTrainingDial = new TrainingForm;
    m_pOutputTable 	= new Output;

    m_pNew          = new QAction(QObject::tr("New project"), 0);
    m_pSave         = new QAction(QObject::tr("Save project"), 0);
    m_pSave->setDisabled(true);
    m_pLoad         = new QAction(QObject::tr("Load project"), 0);
    m_pQuit         = new QAction(QObject::tr("Close project"), 0);

    m_pZoomIn 		= new QAction(QObject::tr("Zoom in"), 0);
    m_pZoomOut 		= new QAction(QObject::tr("Zoom out"), 0);
    m_pShowEdges 	= new QAction(QObject::tr("Show edges"), 0);
    m_pShowEdges->setCheckable(true);
    m_pShowEdges->setChecked(true);
    m_pShowEdges->setDisabled(true);
    m_pShowNodes 	= new QAction(tr("Show nodes"), 0);
    m_pShowNodes->setCheckable(true);
    m_pShowNodes->setChecked(true);
    m_pShowNodes->setDisabled(true);

    setCentralWidget(m_pTabBar);
    addToolBar(Qt::RightToolBarArea, m_ActionsBar);

    createTabs();
    createMenus();
    createActions();
    createGraph();

    connect(m_pViewer->getScene(), SIGNAL(si_netChanged(ANN::BPNet *)), m_pInputDial, SLOT(sl_createTables(ANN::BPNet *)) );
    connect(m_pTabBar, SIGNAL(currentChanged(int)), this, SLOT(sl_tabChanged(int)) );
    connect(m_pInputDial, SIGNAL(si_contentChanged()), this, SLOT(sl_setTrainingSet()) );
}

void MainWindow::sl_tabChanged(int iTab) {
	if(iTab == 1) {
		// Resize table widgets
		m_pInputDial->sl_createTables(m_pANNet);
	 	// Reload the IO widget
		m_pInputDial->setTrainingSet(m_pTrainingSet);
	}
}

MainWindow::~MainWindow()
{
    
}

void MainWindow::createGraph() {
	// give the axes some labels:
	m_pCustomPlot->xAxis->setLabel(QObject::tr("Training cycle (t)") );
	m_pCustomPlot->yAxis->setLabel(QObject::tr("Standard Deviation (SE)") );
	// set axes ranges, so we see all data:
	m_pCustomPlot->xAxis->setRange(0, 1);
	m_pCustomPlot->yAxis->setRange(0, 10);
}

void MainWindow::createTabs() {
    m_pTabBar->insertTab(0, m_pViewer, QIcon("gfx/monitor_icon.png"), QObject::tr("Designer") );
    m_pTabBar->setTabEnabled(0, true);
    m_pTabBar->insertTab(1, m_pInputDial, QIcon("gfx/training_icon.png"), QObject::tr("Input/Output") );
    m_pTabBar->setTabEnabled(1, true);
    m_pTabBar->insertTab(2, m_pTrainingDial, QIcon("gfx/QuestionMark.png"), QObject::tr("Configuration") );
    m_pTabBar->setTabEnabled(2, true);
    m_pTabBar->insertTab(3, m_pCustomPlot, QIcon("gfx/graph_icon.png"), QObject::tr("Learning curve") );
    m_pTabBar->setTabEnabled(3, true);
    m_pTabBar->insertTab(4, m_pOutputTable, QIcon("gfx/output_icon.png"), QObject::tr("Output data") );
    m_pTabBar->setTabEnabled(4, true);

    m_pTabBar->setCurrentIndex(0);
    m_pTabBar->addCornerWidget(m_pActionBar);
}

void MainWindow::createMenus() {
    m_pFileMenu = menuBar()->addMenu(tr("&File"));
    m_pFileMenu->addAction(m_pNew);
    m_pFileMenu->addSeparator();
    m_pFileMenu->addAction(m_pSave);
    m_pFileMenu->addAction(m_pLoad);
    m_pFileMenu->addSeparator();
    m_pFileMenu->addAction(m_pQuit);

    m_pViewMenu = menuBar()->addMenu(tr("&View"));
    m_pViewMenu->addAction(m_pZoomIn);
    m_pViewMenu->addAction(m_pZoomOut);
    m_pViewMenu->addSeparator();
    m_pViewMenu->addAction(m_pShowEdges);
    m_pViewMenu->addAction(m_pShowNodes);

    connect(m_pNew, SIGNAL(triggered ()), this, SLOT(sl_newProject()) );
    connect(m_pSave, SIGNAL(triggered ()), this, SLOT(sl_saveANNet()) );
    connect(m_pLoad, SIGNAL(triggered ()), this, SLOT(sl_loadANNet()) );
    connect(m_pQuit, SIGNAL(triggered ()), this, SLOT(close()) );

    connect(m_pZoomIn, SIGNAL(triggered ()), this, SLOT(sl_zoomIn()) );
    connect(m_pZoomOut, SIGNAL(triggered ()), this, SLOT(sl_zoomOut()) );
    connect(m_pShowEdges, SIGNAL(toggled (bool)), this, SLOT(sl_ShowEdges(bool)) );
    connect(m_pShowNodes, SIGNAL(toggled (bool)), this, SLOT(sl_ShowNodes(bool)) );
}

void MainWindow::sl_zoomIn() {
	double scaleFactor = 1.15; //How fast we zoom
	m_pViewer->scale(scaleFactor, scaleFactor);
}

void MainWindow::sl_zoomOut() {
	double scaleFactor = 1.15; //How fast we zoom
	m_pViewer->scale(1.f/scaleFactor, 1.f/scaleFactor);
}

void MainWindow::sl_ShowEdges(bool bState) {
	foreach(Edge *pEdge, m_pViewer->getScene()->edges() ) {
		pEdge->setVisible(bState);
	}
}

void MainWindow::sl_ShowNodes(bool bState) {
	foreach(Node *pNode, m_pViewer->getScene()->nodes() ) {
		pNode->setVisible(bState);
	}
}

void MainWindow::sl_newProject() {
    m_pShowEdges->setCheckable(true);
    m_pShowEdges->setChecked(true);
    m_pShowEdges->setDisabled(true);

    m_pShowNodes->setCheckable(true);
    m_pShowNodes->setChecked(true);
    m_pShowNodes->setDisabled(true);

    m_pSave->setDisabled(true);
    m_pANNet 		= NULL;
    m_pTrainingSet 	= NULL;

    m_pViewer->getScene()->clearAll();

	m_pRunInput->setDisabled(true);
	m_pStartTraining->setDisabled(true);

	// Reset tables
	m_pOutputTable->reset();
	m_pInputDial->reset();
}

void MainWindow::sl_saveANNet() {
	if(m_pANNet) {
		QString fileName = QFileDialog::getSaveFileName(this, QObject::tr("Save file"), "/home/", QObject::tr("ANNet Files (*.annet)") );
		m_pANNet->ExpToFS(fileName.toStdString() );
	}
}

void MainWindow::sl_loadANNet() {
    m_pShowEdges->setCheckable(true);
    m_pShowEdges->setChecked(true);
    m_pShowEdges->setDisabled(false);

    m_pShowNodes->setCheckable(true);
    m_pShowNodes->setChecked(true);
    m_pShowNodes->setDisabled(false);

	QString fileName = QFileDialog::getOpenFileName(this, QObject::tr("Open file"), "/home/", QObject::tr("ANNet Files (*.annet)") );
	if(fileName != "" && fileName.contains(".annet")) {
		// Remove all of the old content from screen
		m_pViewer->getScene()->clearAll();
		// Create a new net in memory
		m_pANNet = new ANN::BPNet;
		m_pANNet->ImpFromFS(fileName.toStdString());
		// Load content from net to the screen
		m_pViewer->getScene()->setANNet(*m_pANNet);

		/*
		 * Load current Training set of the net
		 */
		if(m_pANNet == NULL) {
			std::cout<<"STRANGE error occurred"<<std::endl;
			assert(m_pANNet != NULL);
			return;
		}

		m_pOutputTable->reset();

		m_pTrainingSet = m_pANNet->GetTrainingSet();
		if(m_pTrainingSet) {
			m_pRunInput->setDisabled(false);
			// Resize table widgets
			m_pInputDial->sl_createTables(m_pANNet);
		 	// Reload the IO widget
			m_pInputDial->setTrainingSet(m_pTrainingSet);
		}
		else {
			m_pRunInput->setDisabled(true);
			// Reset tables
			m_pInputDial->reset();
			// Resize table widgets
			m_pInputDial->sl_createTables(m_pANNet);
		}
	}
}

void MainWindow::createActions() {
    QIcon iconLayer("gfx/layer.png");
    QIcon iconNeuron("gfx/neuron.png");
    QIcon iconEdge("gfx/edge.png");

    QIcon iconRemNeuron("gfx/rem_neuron.png");
    QIcon iconRemLayer("gfx/rem_layer.png");

    QIcon iconRemEdge("gfx/rem_edge.png");
    QIcon iconRemEdges("gfx/rem_edges.png");

    QIcon iconStartTraining("gfx/train.png");
    QIcon iconRun("gfx/run.png");
    QIcon iconBuild("gfx/build.png");

    /*
     * Fancy action bar
     */
    m_pBuildNet = new QAction(iconBuild, QObject::tr("Run through input"), 0);
    m_pActionBar->insertAction(0, m_pBuildNet);
    m_pBuildNet->setDisabled(false);

    m_pStartTraining = new QAction(iconStartTraining, QObject::tr("Start Training"), 0);
    m_pActionBar->insertAction(1, m_pStartTraining);
    m_pStartTraining->setDisabled(true);

    m_pRunInput = new QAction(iconRun, QObject::tr("Run through input"), 0);
    m_pActionBar->insertAction(2, m_pRunInput);
    m_pRunInput->setDisabled(true);

    connect(m_pStartTraining, SIGNAL(triggered ()), this, SLOT(sl_startTraining()) );
    connect(m_pRunInput, SIGNAL(triggered ()), this, SLOT(sl_run()) );
    connect(m_pBuildNet, SIGNAL(triggered ()), this, SLOT(sl_build()) );

    /*
     * Regular tool bar
     */
    m_pAddLayer = m_ActionsBar->addAction(iconLayer, "Add a layer");
    m_pRemoveLayers = m_ActionsBar->addAction(iconRemLayer, "Remove selected layers");
    m_ActionsBar->addSeparator();
    m_pAddNeuron = m_ActionsBar->addAction(iconNeuron, "Add neurons to selected layers");
    m_pRemoveNeurons = m_ActionsBar->addAction(iconRemNeuron, "Remove selected neurons");
    m_ActionsBar->addSeparator();
    m_pAddEdges = m_ActionsBar->addAction(iconEdge, "Add edges to selected neurons");
    m_pRemoveEdges = m_ActionsBar->addAction(iconRemEdge, "Remove selected edges");
    m_ActionsBar->addSeparator();
    m_pRemoveAllEdges = m_ActionsBar->addAction(iconRemEdges, "Remove all edges");
    m_ActionsBar->addSeparator();
    m_pSetTrainingPairs = m_ActionsBar->addAction(iconRemEdges, "Set number of training pairs");

    connect(m_pAddLayer, SIGNAL(triggered ()), this, SLOT(sl_createLayer()) );
    connect(m_pAddNeuron, SIGNAL(triggered ()), m_pViewer, SLOT(sl_addNeurons()) );
    connect(m_pAddEdges, SIGNAL(triggered ()), m_pViewer, SLOT(sl_createConnections()) );

    connect(m_pRemoveLayers, SIGNAL(triggered ()), m_pViewer, SLOT(sl_removeLayers()) );
    connect(m_pRemoveNeurons, SIGNAL(triggered() ), m_pViewer, SLOT(sl_removeNeurons()) );

    connect(m_pRemoveEdges, SIGNAL(triggered ()), m_pViewer, SLOT(sl_removeConnections()) );
    connect(m_pRemoveAllEdges, SIGNAL(triggered() ), m_pViewer, SLOT(sl_removeAllConnections()) );

    connect(m_pSetTrainingPairs, SIGNAL(triggered() ), m_pInputDial, SLOT(sl_setNmbrOfSets()) );
}


void MainWindow::sl_build() {
	m_pANNet = m_pViewer->getScene()->getANNet();
}

void MainWindow::sl_run() {
	if(m_pANNet && m_pTrainingSet) {
		m_pANNet->SetTrainingSet(m_pTrainingSet);
		m_pOutputTable->display(m_pANNet);
	}
}

void MainWindow::sl_setTrainingSet() {
	m_pRunInput->setDisabled(false);
	m_pStartTraining->setDisabled(false);

	if(m_pInputDial->getTrainingSet())
		m_pTrainingSet = m_pInputDial->getTrainingSet();
}

void MainWindow::sl_startTraining() {
	int iCycles 			= m_pTrainingDial->getMaxCycles();
	float fMaxError 		= m_pTrainingDial->getMaxError();
	float fLearningRate 	= m_pTrainingDial->getLearningRate();
	float fMomentum 		= m_pTrainingDial->getMomentum();
	float fWeightDecay 		= m_pTrainingDial->getWeightDecay();
 	std::string sTFunct 	= m_pTrainingDial->getTransfFunct().data();

 	// Save current training set
	m_pTrainingSet 			= m_pInputDial->getTrainingSet();
	// Get current net
	if(m_pANNet == NULL) {
		m_pANNet 			= m_pViewer->getScene()->getANNet();
	}
 	// Reload the IO widget
	m_pInputDial->setTrainingSet(m_pTrainingSet);

	if(m_pANNet == NULL) {
		m_pSave->setDisabled(true);
		return;
	}
	else {
		m_vErrors.clear();
		m_pSave->setDisabled(false);

		m_pANNet->SetLearningRate(fLearningRate);
		m_pANNet->SetMomentum(fMomentum);
		m_pANNet->SetWeightDecay(fWeightDecay);
		m_pANNet->SetTransfFunction(ANN::Functions::ResolveTransfFByName(sTFunct.data()));

		m_pANNet->SetTrainingSet(m_pTrainingSet);
		m_vErrors = m_pANNet->TrainFromData(iCycles, 0.001);
		iCycles = m_vErrors.size();

		sl_run();

		// generate some data to plot:
		float fGreatest = m_vErrors[0];
		QVector<double> x(iCycles), y(iCycles); // initialize with entries 0..100
		for (int i=0; i < iCycles; i++) {
			x[i] = i;
			y[i] = m_vErrors[i];
			if(fGreatest < m_vErrors[i])
				fGreatest = m_vErrors[i];
		}

		// set axes ranges, so we see all data:
		m_pCustomPlot->xAxis->setRange(0, iCycles);
		m_pCustomPlot->yAxis->setRange(0, fGreatest);
		// create graph and assign data to it:
		m_pCustomPlot->addGraph();
		m_pCustomPlot->graph(0)->setData(x, y);
		m_pCustomPlot->graph(0)->setBrush(QBrush(QColor(0, 0, 255, 20))); // first graph will be filled with translucent blue
		m_pCustomPlot->replot();
	}
}

void MainWindow::sl_createLayer() {
    m_pShowEdges->setDisabled(false);
    m_pShowNodes->setDisabled(false);

    QPointF pCenter = m_pViewer->getScene()->sceneRect().center();
    Layer *pLayer = m_pViewer->getScene()->addLayer(1, pCenter, "no type");
}
