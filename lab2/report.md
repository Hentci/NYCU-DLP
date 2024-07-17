B. SCCNet

This section presents the design of the proposed SCCNet and its architecture. A major part of SCCNet consists of convolutional kernels that capture the spatial and temporal characteristics of the EEG data. The design of SCCNet focuses on leveraging the benefits from applying spatial filtering to EEG data for purposes such as feature extraction and noise suppression [5], [6]. The architecture of SCCNet is illustrated in Fig. 1. The input to SCCNet is multi-channel EEG data arranged in 2-dimensions, with C channels and T time points. The architecture of SCCNet consists of four blocks: the first convolution block, the second convolution block, the pooling block, and the softmax block. The SCCNet herein is implemented using the Keras platform [13].

In the first and second blocks, the SCCNet performs two-step 2-dimensional convolution procedures. The initial convolution extracts EEG features, mimicking a spatial com- ponent analysis that decomposes the original EEG data from the channel domain to a component domain, where Nu filters with a kernel size of (Ｃ, Nt). When Nt = 1, this convolution step essentially performs a linear combination of EEG signals across all channels. This procedure is referred to conven- tional spatial filtering or component analysis technique that is commonly used for signal augmentation, noise reduction, and/or artifact removal [5]. Since the multi-channel EEG data are transferred into multiple EEG spatial components, the size of channel domain becomes 1 and the size of component domain becomes the number of convolutional kernels, Nu. Thus, the tensor dimension is permuted in an order of (2,1,3) to switch the second dimension to the first for the second convolutional block.

The second convolution uses Nc convolutional kernels with a size of (Nu, 12), where the number 12 corresponds to 0.1 seconds along time domain. At this step, the convolution procedure applies to both the temporal and spatial component domains. The spatial-temporal convolution is expected to perform spectral filtering, inter-component correlation and other spatial-temporal analysis on the EEG spatial compo- nents. Zero-padding and batch normalization were applied to both the first and second convolutions, as well as l2 regularization with a coefficient of 0.0001. We use square activation to extract the power from the data, as spectral power change is the most prominent marker in MI EEG. Next, we applied dropout with a rate of 0.5 to prevent over- fitting.



Following the two convolutional blocks, we apply an average pooling layer of size (1,62) to perform smoothing in the temporal domain and reduce the dimension, where the number 62 corresponds to 0.5 seconds along time domain. The final block performs a softmax classification with 4 units corresponding to the 4 classes in the MI task: left hand, right hand, feet, and tongue.



SCCNet

這一部分介紹了提出的 SCCNet 的設計及其架構。SCCNet 的主要部分由卷積核組成，用於捕捉 EEG 數據的空間和時間特徵。SCCNet 的設計重點在於利用空間濾波來進行特徵提取和噪聲抑制。SCCNet 的架構如圖 1 所示。SCCNet 的輸入是排列成 2 維的多通道 EEG 數據，具有  C  個通道和  T  個時間點。SCCNet 的架構由四個模塊組成：第一卷積塊、第二卷積塊、池化塊和 softmax 分類塊。本文中的 SCCNet 使用 Keras 平台實現。

第一卷積塊與第二卷積塊

在第一和第二卷積塊中，SCCNet 進行兩步 2 維卷積操作。初始卷積提取 EEG 特徵，模擬空間成分分析，將原始 EEG 數據從通道域分解到成分域，其中包含  Nu  個大小為  (C, Nt)  的濾波器。當  Nt = 1  時，這一步卷積本質上對所有通道的 EEG 信號進行線性組合。這一過程被稱為傳統的空間濾波或成分分析技術，常用於信號增強、噪聲減少和/或偽影去除。由於多通道 EEG 數據被轉換為多個 EEG 空間成分，通道域的大小變為 1，而成分域的大小變為卷積核的數量  Nu 。因此，張量的維度按順序  (2,1,3)  進行排列，以便將第二維度切換為第一卷積塊的輸入。

第二次卷積使用大小為  (Nu, 12)  的  Nc  個卷積核，其中數字 12 對應於時間域的 0.1 秒。在這一步驟中，卷積過程同時應用於時間和空間成分域。預期的空間-時間卷積將執行光譜過濾、成分間相關及其他空間-時間分析。對於第一和第二卷積，均應用了零填充和批量正規化，以及係數為 0.0001 的  l2  正則化。我們使用平方激活來提取數據的功率，因為光譜功率變化是 MI EEG 中最顯著的標記。接著，我們應用了 0.5 的 dropout 率以防止過擬合。

池化層與分類

在兩個卷積塊之後，我們應用大小為  (1,62)  的平均池化層，在時間域進行平滑並減少維度，其中數字 62 對應於時間域的 0.5 秒。最後一個塊進行 softmax 分類，具有 4 個單元，對應於 MI 任務中的 4 個類別：左手、右手、雙腳和舌頭。

總結

SCCNet 的架構設計旨在通過多層卷積操作來提取 EEG 數據中的空間和時間特徵，並通過批量正規化和 dropout 技術來增強模型的穩健性和防止過擬合。最終的 softmax 分類層能夠將處理後的特徵映射到具體的運動想像任務類別上。