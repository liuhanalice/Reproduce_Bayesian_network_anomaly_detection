%%
clc;
clear all;
colordef white
addpath './npy-matlab-master\npy-matlab';
%% An example of groundtruth
ptCloud = pcread('./EM_DRG_result/Noise_0p10/data3/pcd_0.pcd'); 
label=readNPY("./EM_DRG_result/Noise_0p10//data3/label_0.npy");
pts=ptCloud.Location;
[defect_row,~]=find(label==1);
[normal_row,~]=find(label==0);
[outlier_row,~]=find(label==2);
Colors=uint8(zeros(size(pts,1),3));
for i=1:size(pts,1)
    if label(i)==2
        Colors(i,:)=[255,0,0];
    end
    if label(i)==1
        Colors(i,:)=[0,0,255];
    end
end
pcdRefSurf=pointCloud(pts(normal_row,:),'Color',Colors(normal_row,:));
gca=pcshow(pcdRefSurf,'MarkerSize',2);
hold on
pcdOutlier=pointCloud(pts(outlier_row,:),'Color',Colors(outlier_row,:));
gca=pcshow(pcdOutlier,'MarkerSize',80);
hold on 
pcdDefect=pointCloud(pts(defect_row,:),'Color',Colors(defect_row,:));
gca=pcshow(pcdDefect,'MarkerSize',40);

%gca2=pcshow(ptCloud,'MarkerSize',1);
set(gca,'FontName','Times New Roman','FontSize',20,'color','w');
set(gca, 'XColor', [0.15 0.15 0.15], 'YColor', [0.15 0.15 0.15], 'ZColor', [0.15 0.15 0.15])
set(gcf,'color','w');
grid off
%% An example of our method
ptCloud = pcread('./EM_DRG_result/Noise_0p10/data3/pcd_0.pcd'); 
label=readNPY("./EM_DRG_result/Noise_0p10/result3/EM_DRG_est_label_0.npy");
pts=ptCloud.Location;
[defect_row,~]=find(label==1);
[normal_row,~]=find(label==0);
[outlier_row,~]=find(label==2);
Colors=uint8(zeros(size(pts,1),3));
for i=1:size(pts,1)
    if label(i)==2
        Colors(i,:)=[255,0,0];
    end
    if label(i)==1
        Colors(i,:)=[0,0,255];
    end
end
pcdRefSurf=pointCloud(pts(normal_row,:),'Color',Colors(normal_row,:));
gca=pcshow(pcdRefSurf,'MarkerSize',2);
hold on
pcdOutlier=pointCloud(pts(outlier_row,:),'Color',Colors(outlier_row,:));
gca=pcshow(pcdOutlier,'MarkerSize',80);
hold on 
pcdDefect=pointCloud(pts(defect_row,:),'Color',Colors(defect_row,:));
gca=pcshow(pcdDefect,'MarkerSize',40);

%gca2=pcshow(ptCloud,'MarkerSize',1);
set(gca,'FontName','Times New Roman','FontSize',20,'color','w');
set(gca, 'XColor', [0.15 0.15 0.15], 'YColor', [0.15 0.15 0.15], 'ZColor', [0.15 0.15 0.15])
set(gcf,'color','w');
grid off
%% An example of J Nondestruct
ptCloud = pcread('./EM_DRG_result/Noise_0p10/data3/pcd_0.pcd'); 
label=readNPY("./EM_DRG_result/Noise_0p10/result3/J_Nondestruct_est_label_0.npy");
pts=ptCloud.Location;
[defect_row,~]=find(label==1);
[normal_row,~]=find(label==0);
[outlier_row,~]=find(label==2);
Colors=uint8(zeros(size(pts,1),3));
for i=1:size(pts,1)
    if label(i)==2
        Colors(i,:)=[255,0,0];
    end
    if label(i)==1
        Colors(i,:)=[0,0,255];
    end
end
pcdRefSurf=pointCloud(pts(normal_row,:),'Color',Colors(normal_row,:));
gca=pcshow(pcdRefSurf,'MarkerSize',2);
hold on
pcdOutlier=pointCloud(pts(outlier_row,:),'Color',Colors(outlier_row,:));
gca=pcshow(pcdOutlier,'MarkerSize',80);
hold on 
pcdDefect=pointCloud(pts(defect_row,:),'Color',Colors(defect_row,:));
gca=pcshow(pcdDefect,'MarkerSize',40);

%gca2=pcshow(ptCloud,'MarkerSize',1);
set(gca,'FontName','Times New Roman','FontSize',20,'color','w');
set(gca, 'XColor', [0.15 0.15 0.15], 'YColor', [0.15 0.15 0.15], 'ZColor', [0.15 0.15 0.15])
set(gcf,'color','w');
grid off
%% An example of IST OPTIC 
ptCloud = pcread('./EM_DRG_result/Noise_0p10/data3/pcd_0.pcd'); 
label=readNPY("./EM_DRG_result/Noise_0p10/result3/IST_OPTIC_est_label_0.npy");
pts=ptCloud.Location;
[defect_row,~]=find(label==1);
[normal_row,~]=find(label==0);
[outlier_row,~]=find(label==2);
Colors=uint8(zeros(size(pts,1),3));
for i=1:size(pts,1)
    if label(i)==2
        Colors(i,:)=[255,0,0];
    end
    if label(i)==1
        Colors(i,:)=[0,0,255];
    end
end
pcdRefSurf=pointCloud(pts(normal_row,:),'Color',Colors(normal_row,:));
gca=pcshow(pcdRefSurf,'MarkerSize',2);
hold on
pcdOutlier=pointCloud(pts(outlier_row,:),'Color',Colors(outlier_row,:));
gca=pcshow(pcdOutlier,'MarkerSize',80);
hold on 
pcdDefect=pointCloud(pts(defect_row,:),'Color',Colors(defect_row,:));
gca=pcshow(pcdDefect,'MarkerSize',40);

%gca2=pcshow(ptCloud,'MarkerSize',1);
set(gca,'FontName','Times New Roman','FontSize',20,'color','w');
set(gca, 'XColor', [0.15 0.15 0.15], 'YColor', [0.15 0.15 0.15], 'ZColor', [0.15 0.15 0.15])
set(gcf,'color','w');
grid off
%% EFA 
ptCloud = pcread('./EM_DRG_result/Noise_0p10/data3/pcd_0.pcd'); 
label=readNPY("./EM_DRG_result/Noise_0p10/result3/EFA_est_label_0.npy");
pts=ptCloud.Location;
[defect_row,~]=find(label==1);
[normal_row,~]=find(label==0);
[outlier_row,~]=find(label==2);
Colors=uint8(zeros(size(pts,1),3));
for i=1:size(pts,1)
    if label(i)==2
        Colors(i,:)=[255,0,0];
    end
    if label(i)==1
        Colors(i,:)=[0,0,255];
    end
end
pcdRefSurf=pointCloud(pts(normal_row,:),'Color',Colors(normal_row,:));
gca=pcshow(pcdRefSurf,'MarkerSize',2);
hold on
pcdOutlier=pointCloud(pts(outlier_row,:),'Color',Colors(outlier_row,:));
gca=pcshow(pcdOutlier,'MarkerSize',80);
hold on 
pcdDefect=pointCloud(pts(defect_row,:),'Color',Colors(defect_row,:));
gca=pcshow(pcdDefect,'MarkerSize',40);

%gca2=pcshow(ptCloud,'MarkerSize',1);
set(gca,'FontName','Times New Roman','FontSize',20,'color','w');
set(gca, 'XColor', [0.15 0.15 0.15], 'YColor', [0.15 0.15 0.15], 'ZColor', [0.15 0.15 0.15])
set(gcf,'color','w');
grid off
