! name list, cioè liste di nomi; desueto, si usa per piccoli programmi e calcoli rapidi. 

program example
    implicit none
    integer :: i, n
    double precision :: x, y !numero reale; vorrei avere un modo rapido perchè il programma legga due valori
                            !x e i da un file di testo
    namelist /inputdata/ i, x
    namelist /outputdata/ y

    open(unit = 10, file="data_in.txt", status="old") !devo aprire un file; ci si riferisce a
                                                    ! un file con un numero intero (old e non unknown,
                                                    !altrimenti lo sovrascrive). Unknown serve ad avere
                                                    !i risultati in un file di output data
    read(10, nml=inputdata) !nml = namelist, input o output, ma non bisogna specificare il formato
    close(10)

    print *, i, x

    y = x*x !l'unità può essere la stessa, l'importante è che non abbia due file aperti insieme con la
            !stessa unità.

    open(unit=10, file="data_out.txt", status="unknown")
    read(10, nml=outputdata) !l'output viene salvato nella stessa cartella dove sta il prompt,
                                !non quella in cui c'è il sorgente (mkdir per spostarlo, rm per rimuoverlo)
                                !la notazione ./ serve a salire di una cartella gerarchicamente;
                                !la notazione ../ serve a salire di due cartelle


!controllo di flusso, come in python, cicli for e while con le condizioni di verità
!per gli operatori booleani si usa == .eq.; != .ne. significa diverso; < .lt. minore; <= .le. minore
!o uguale; idem per il maggiore.

    if (2 .gt. 3) then !al posto di :
        print *, "A"!l'indentazione non è fissata in fortran
    else if (2 .gt. 1) then
        print *, "B"
!è utile mettere una condizione di default se nessuna delle condizioni si avvera
    else
        print *, "C"
    end if !è importante concludere il ciclo if

!il corrispettivo del ciclo for è il ciclo do
    do i = 1, 3
        print *, i
    end do

    i = 1
    do while (i .le. 3)
        print *, i
        i = i + 1 !si tratta di due cose temporalmente distinte, significa i +1, calcolare e assegnarlo a i
    end do
!per dare la dimensione ad un array bisogna allocarlo
    n = 3 !dimensione dell'array
    allocate(v(n))

!per inizializzare gli elementi di un array, si può fare (/ 1, 2, 3 /). se è grande, si può inizializzare
    !con un ciclo

    do i = 1, n
        v(i) = 2*i !assegno un valore all'elemento i esimo di un array
    end do

    print *, v
!la cosa si può fare in maniera molto più compatta con un loop

    v = (/ (2*1, i = 1, n)/) !(/ per indicare che inizia un array, poi (per valore implicito, range)/)
    print *, (v(i), i = 1, n) !si tratta di un ciclo implicito

!rivedi parte sui sys --> ricopia

end program example